// Changed selector only. Actual generated ABI; independent CPU full-sort oracle.
// debug2 consumes score bits, never used by public Search or its performance path.
#include "fused_runtime.h"
#include <numeric>
#if !defined(BM25_SIMD_CHECK_SHA)
#error Build using the isolated selector-check CMake project.
#endif
namespace {
uint32_t Key(uint32_t bits) { return ~((bits&0x7fffffffU)==0 ? 0U : bits); }
struct Api {
    ApiLibrary library;
    FusedPrepare prepare;
    Execute execute;
    explicit Api(const fs::path &vendors): library(vendors/"bm25_fused_score_topk") {
        prepare=library.Symbol<FusedPrepare>("aclnnBm25FusedScoreTopKGetWorkspaceSize");
        execute=library.Symbol<Execute>("aclnnBm25FusedScoreTopK");
    }
};
}
int main(int argc,char **argv) {
    try {
        Check(argc==4,"Usage: selector_check VENDORS DEVICE NEW_OUTPUT_DIRECTORY");
        const auto output=fs::absolute(argv[3]); Check(!fs::exists(output),"Output exists");
        Api api(fs::absolute(argv[1])); Runtime runtime;
        runtime.device=static_cast<int>(Number(argv[2],0,INT32_MAX));
        Require(aclInit(nullptr),"aclInit");runtime.initialized=true;
        Require(aclrtSetDevice(runtime.device),"set device");runtime.selected=true;
        Require(aclrtCreateContext(&runtime.context,runtime.device),"context");
        Require(aclrtCreateStream(&runtime.stream),"stream");
        size_t checked=0;
        for(const size_t width:{33,65,1025,2049,4097,8192}) for(const size_t k:{1,10,32}) {
            std::vector<size_t> counts={0,1,k-1,k,k+1,63,64,65,511,512,513,1023,1024,1025,width};
            for(auto &n:counts)n=std::min(n,width);
            std::sort(counts.begin(),counts.end());counts.erase(std::unique(counts.begin(),counts.end()),counts.end());
            const size_t q=counts.size()*2, size=q*width, selected=q*k, guard=16;
            Buffers buffers;buffers.stream=runtime.stream;Tensors tensors;tensors.stream=runtime.stream;
            std::vector<int64_t> termOffsets={0,static_cast<int64_t>(size)}, offsets(q+1);
            std::iota(offsets.begin(),offsets.end(),0);
            std::vector<int32_t> docs(size,0),words(q,0);
            std::vector<float> norm(width,1),idf(1,1);
            const void *host[6]={termOffsets.data(),docs.data(),norm.data(),idf.data(),offsets.data(),words.data()};
            const size_t num[6]={2,size,width,1,q+1,q}, bytes[6]={8,4,4,4,8,4};
            const aclDataType type[6]={ACL_INT64,ACL_INT32,ACL_FLOAT,ACL_FLOAT,ACL_INT64,ACL_INT32};
            aclTensor *in[6];
            for(size_t i=0;i<6;++i){
                void *mem=buffers.Allocate(Bytes(num[i],bytes[i]));
                Require(aclrtMemcpy(mem,Bytes(num[i],bytes[i]),host[i],Bytes(num[i],bytes[i]),ACL_MEMCPY_HOST_TO_DEVICE),"input");
                in[i]=tensors.Describe1D(num[i],type[i],mem);
            }
            void *input=buffers.Allocate(Bytes(size,4));
            auto *tf=tensors.Describe1D(size,ACL_FLOAT,input);
            void *out[3]={buffers.Allocate(Bytes(selected+guard,4)),buffers.Allocate(Bytes(selected+guard,4)),
                buffers.Allocate(Bytes(size+guard,4))};
            auto *scores=Describe3D(tensors,q,1,k,ACL_FLOAT,out[0]);
            auto *ids=Describe3D(tensors,q,1,k,ACL_INT32,out[1]);
            auto *debug=tensors.Describe(q,width,ACL_FLOAT,out[2]);
            uint64_t capacity=0;void *workspace=nullptr;
            for(size_t repeat=0;repeat<3;++repeat) {
                std::vector<uint32_t> bits(size);
                for(size_t row=0;row<q;++row) {
                    for(size_t d=0;d<width;++d)bits[row*width+d]=d%2?0x80000000U:0;
                    for(size_t n=0;n<counts[row/2];++n) {
                        const size_t d=(n*19+(repeat==1?13:0)+row)%width;
                        // gcd(19,width)=1 for all listed widths. A/B/A real-ID positions.
                        const uint32_t varied[]={1U,2U,0x007fffffU,0x00800000U,0x3f7fffffU,
                            0x3f800000U,0x3f800001U,0x40000000U,0x7f7fffffU,0x00800001U};
                        bits[row*width+d]=row%2?varied[n%10]:0x3f800000U;
                    }
                }
                Require(aclrtMemcpy(input,Bytes(size,4),bits.data(),Bytes(size,4),ACL_MEMCPY_HOST_TO_DEVICE),"score bits");
                std::vector<uint32_t> poison(size+guard,UINT32_MAX);
                for(size_t i=0;i<3;++i){const size_t n=(i==2?size:selected)+guard;
                    Require(aclrtMemcpy(out[i],Bytes(n,4),poison.data(),Bytes(n,4),ACL_MEMCPY_HOST_TO_DEVICE),"poison");}
                uint64_t need=0;aclOpExecutor *executor=nullptr;
                Require(api.prepare(in[0],in[1],tf,in[2],in[3],in[4],in[5],k,1,2,
                    scores,ids,debug,&need,&executor),"prepare selector debug2");
                Check(executor!=nullptr,"No executor");
                if(repeat==0){capacity=need;workspace=buffers.Allocate(need);}
                Check(need<=capacity,"Workspace capacity changed");
                Require(api.execute(workspace,need,executor,runtime.stream),"execute selector debug2");
                Require(aclrtSynchronizeStream(runtime.stream),"sync");
                std::vector<uint32_t> actual[3];
                for(size_t i=0;i<3;++i){const size_t n=(i==2?size:selected)+guard;actual[i].resize(n);
                    Require(aclrtMemcpy(actual[i].data(),Bytes(n,4),out[i],Bytes(n,4),ACL_MEMCPY_DEVICE_TO_HOST),"download");
                    for(size_t j=n-guard;j<n;++j)Check(actual[i][j]==UINT32_MAX,"Guard overwritten");}
                Check(std::equal(bits.begin(),bits.end(),actual[2].begin()),"Input/debug score bits changed");
                for(size_t row=0;row<q;++row){
                    std::vector<uint32_t> oracle(width);std::iota(oracle.begin(),oracle.end(),0);
                    std::sort(oracle.begin(),oracle.end(),[&](uint32_t a,uint32_t b){
                        const auto ka=Key(bits[row*width+a]),kb=Key(bits[row*width+b]);return ka<kb||(ka==kb&&a<b);});
                    for(size_t rank=0;rank<k;++rank){const size_t at=row*k+rank;const auto id=oracle[rank];
                        Check(actual[1][at]==id && actual[0][at]==bits[row*width+id],"First selector ID/FP32 mismatch");}
                    ++checked;
                }
            }
            Check(tensors.Close()&&buffers.Close(),"Cleanup failed");
            std::cout<<"selector_checked_width="<<width<<" k="<<k<<" calls=3"<<std::endl;
        }
        Check(runtime.Close(),"Runtime cleanup failed");Check(fs::create_directories(output),"New output required");
        std::ofstream f(output/"report.json");
        f<<"{\"status\":\"actual_simd_selector_guards_and_strict_bit_rankings_passed\",\"checked_rows\":"<<checked
            <<",\"calls\":54,\"host_sha256\":\""<<BM25_SIMD_CHECK_SHA<<"\"}\n";
        f.close();Check(bool(f),"Report write failed");return 0;
    }catch(const std::exception &e){std::cerr<<"FAILED: "<<e.what()<<std::endl;return 1;}
}
