// Diagnostic only: exact accepted partial/global outputs plus raw device-stage clocks.
#include "fused_runtime.h"
#include "../../common/shard_directory.h"

struct DiagnosticApi {
    ApiLibrary fused,merge;
    FusedPrepare prepare; MergePrepare prepareMerge; Execute execute,executeMerge;
    explicit DiagnosticApi(const fs::path &vendors):fused(vendors/"bm25_fused_score_topk"),merge(vendors/"bm25_topk_merge") {
        prepare=fused.Symbol<FusedPrepare>("aclnnBm25FusedScoreTopKGetWorkspaceSize");
        execute=fused.Symbol<Execute>("aclnnBm25FusedScoreTopK");
        prepareMerge=merge.Symbol<MergePrepare>("aclnnBm25TopKMergeGetWorkspaceSize");
        executeMerge=merge.Symbol<Execute>("aclnnBm25TopKMerge");
    }
};

int main(int argc,char **argv) {
    try {
        Check(argc==7,"Usage: STAGES INDEX REQUESTS ACCEPTED_ACTUAL VENDORS DEVICE NEW_OUTPUT");
        const Index index(fs::absolute(argv[1])); auto requests=LoadRequests(fs::absolute(argv[2]),index.norm.size());
        Check(requests.size()==2 && requests[0].label=="A" && requests[1].label=="B","Expected same accepted A/B");
        const auto reference=fs::absolute(argv[3]),output=fs::absolute(argv[6]);
        Check(!fs::exists(output),"Diagnostic output exists");
        DiagnosticApi api(fs::absolute(argv[4]));Runtime runtime;
        runtime.device=static_cast<int>(Number(argv[5],0,INT32_MAX));
        Require(aclInit(nullptr),"init");runtime.initialized=true;
        Require(aclrtSetDevice(runtime.device),"device");runtime.selected=true;
        Require(aclrtCreateContext(&runtime.context,runtime.device),"context");
        Require(aclrtCreateStream(&runtime.stream),"stream");
        Buffers buffers;Tensors tensors;buffers.stream=tensors.stream=runtime.stream;
        constexpr size_t q=8,shards=32,k=10,guard=16;
        const size_t n=index.norm.size(),selected=q*shards*k,top=q*k,clockWords=q*shards*16;
        Check(n==262144 && clockWords<=q*n,"Expected accepted256K/Q8 and sufficient diagnostic output");
        const auto directory=bm25plan::BuildDirectory(index.offsets,index.docs,n,shards);
        const void *host[5]={directory.data(),index.docs.data(),index.tf.data(),index.norm.data(),index.idf.data()};
        const size_t lengths[5]={directory.size(),index.docs.size(),index.tf.size(),n,index.idf.size()};
        const aclDataType types[5]={ACL_INT64,ACL_INT32,ACL_FLOAT,ACL_FLOAT,ACL_FLOAT};
        aclTensor *inputs[5];
        for(size_t i=0;i<5;++i) {
            const size_t bytes=Bytes(lengths[i],i?4:8);void *p=buffers.Allocate(bytes);
            Require(aclrtMemcpy(p,bytes,host[i],bytes,ACL_MEMCPY_HOST_TO_DEVICE),"index upload");
            inputs[i]=tensors.Describe1D(lengths[i],types[i],p);
        }
        void *offsets=buffers.Allocate(Bytes(q+1,8)),*words=buffers.Allocate(Bytes(24,4));
        auto *offsetTensor=tensors.Describe1D(q+1,ACL_INT64,offsets),*wordTensor=tensors.Describe1D(24,ACL_INT32,words);
        void *raw[5]; const size_t sizes[5]={selected,selected,top,top,q*n};
        aclTensor *out[5];
        for(size_t i=0;i<5;++i) {
            raw[i]=buffers.Allocate(Bytes(sizes[i]+2*guard,4));auto *body=static_cast<uint32_t *>(raw[i])+guard;
            out[i]=i<2?Describe3D(tensors,q,shards,k,i==0?ACL_FLOAT:ACL_INT32,body):
                tensors.Describe(q,i==4?n:k,i==3?ACL_INT32:ACL_FLOAT,body);
        }
        Check(fs::create_directories(output),"Cannot create new diagnostic output");
        size_t calls=0;
        for(size_t cycle=0;cycle<15;++cycle) for(size_t slot=0;slot<2;++slot) {
            auto &request=requests[slot^(cycle&1)];
            Check(request.Queries()==q && request.terms.size()<=24,"Accepted fixed query packet differs");
            std::vector<int32_t> padded(24,0);std::copy(request.terms.begin(),request.terms.end(),padded.begin());
            Require(aclrtMemcpy(offsets,Bytes(q+1,8),request.offsets.data(),Bytes(q+1,8),ACL_MEMCPY_HOST_TO_DEVICE),"query offsets");
            Require(aclrtMemcpy(words,Bytes(24,4),padded.data(),Bytes(24,4),ACL_MEMCPY_HOST_TO_DEVICE),"query words");
            for(size_t i=0;i<5;++i) {
                const std::vector<uint32_t> poison(sizes[i]+2*guard,0xa5a5a5a5U);
                Require(aclrtMemcpy(raw[i],Bytes(poison.size(),4),poison.data(),Bytes(poison.size(),4),ACL_MEMCPY_HOST_TO_DEVICE),"poison output and guards");
            }
            uint64_t bytes[2]{};aclOpExecutor *executors[2]{};
            Require(api.prepare(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],offsetTensor,wordTensor,k,shards,3,
                out[0],out[1],out[4],&bytes[0],&executors[0]),"prepare stage diagnostic");
            Require(api.prepareMerge(out[0],out[1],k,n,out[2],out[3],&bytes[1],&executors[1]),"prepare diagnostic merge");
            // Diagnostic executors may be consumed once. Workspace is freed after stream completion.
            Buffers work;work.stream=runtime.stream;
            for(size_t i=0;i<2;++i)Check(executors[i]!=nullptr,"Missing executor");
            Require(api.execute(work.Allocate(bytes[0]),bytes[0],executors[0],runtime.stream),"execute stages");
            Require(api.executeMerge(work.Allocate(bytes[1]),bytes[1],executors[1],runtime.stream),"execute merge");
            Require(aclrtSynchronizeStream(runtime.stream),"diagnostic sync");
            const char *names[4]={"partial_scores.bin","partial_ids.bin","top_scores.bin","top_ids.bin"};
            std::vector<uint32_t> downloaded[5];
            for(size_t i=0;i<5;++i) {
                auto &data=downloaded[i];data.resize(sizes[i]+2*guard);
                Require(aclrtMemcpy(data.data(),Bytes(data.size(),4),raw[i],Bytes(data.size(),4),ACL_MEMCPY_DEVICE_TO_HOST),"diagnostic download");
                for(size_t j=0;j<guard;++j)Check(data[j]==0xa5a5a5a5U && data[data.size()-1-j]==0xa5a5a5a5U,"Guard overwritten");
                if(i<4) {
                    const auto expected=Read<uint32_t>(reference/("before_"+request.label)/names[i]);
                    Check(expected.size()==sizes[i] && std::equal(expected.begin(),expected.end(),data.begin()+guard),"Accepted output bits differ");
                } else {
                    Check(std::all_of(data.begin()+guard+clockWords,data.end()-guard,[](uint32_t v){return v==0xa5a5a5a5U;}),"Unexpected full-score write in diagnostic");
                }
            }
            Check(work.Close(),"Workspace cleanup failed");
            if(cycle>=5) {
                std::vector<uint64_t> clocks(q*shards*8);
                std::memcpy(clocks.data(),downloaded[4].data()+guard,Bytes(clocks.size(),8));
                Write(output/(request.label+"_"+std::to_string(cycle-5)+".bin"),clocks);
            }
            ++calls;
        }
        Check(tensors.Close()&&buffers.Close()&&runtime.Close(),"Cleanup failed");
        std::ofstream report(output/"report.json");
        report<<"{\"status\":\"stage_records_outputs_and_guards_checked\",\"calls\":"<<calls
            <<",\"measured_calls\":20,\"q\":8,\"shards\":32,\"host_sha256\":\""<<BM25_STAGE_HOST_SHA<<"\"}\n";
        report.close();Check(bool(report),"Report write failed");return 0;
    } catch(const std::exception &error) {std::cerr<<"FAILED: "<<error.what()<<std::endl;return 1;}
}
