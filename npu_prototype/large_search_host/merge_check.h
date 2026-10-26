#pragma once
// Affected operator checks only; generated finite FP32 bits, strict global IDs and guards.
inline void CheckLargeMerge(const fs::path &vendors,int device,const fs::path &output) {
    Check(!fs::exists(output),"New merge-check output required");SearchFusedApi api(vendors);Runtime runtime;
    runtime.device=device;Require(aclInit(nullptr),"init merge check");runtime.initialized=true;
    Require(aclrtSetDevice(device),"merge check device");runtime.selected=true;
    Require(aclrtCreateContext(&runtime.context,device),"merge context");Require(aclrtCreateStream(&runtime.stream),"merge stream");
    uint64_t calls=0,checked=0;
    for(uint64_t leaves:{1ULL,31ULL,32ULL,33ULL,1023ULL,1024ULL,1025ULL})for(uint64_t k:{1ULL,10ULL,32ULL}) {
        constexpr uint64_t q=2,guard=16;uint64_t count=leaves;
        std::vector<Candidate> input(q*leaves*k);
        for(uint64_t query=0;query<q;++query)for(uint64_t s=0;s<leaves;++s) {
            auto begin=input.begin()+(query*leaves+s)*k;
            for(uint64_t rank=0;rank<k;++rank) {
                const int32_t id=static_cast<int32_t>(s*k+rank);uint32_t bits=0x3f800000U+((static_cast<uint32_t>(id)*977U)%65536U);
                float value;std::memcpy(&value,&bits,4);begin[rank]={query?value:1.0F,id};
            }
            std::sort(begin,begin+k,Better{});
        }
        Buffers buffers;Tensors tensors;buffers.stream=tensors.stream=runtime.stream;
        auto allocate=[&](uint64_t n,aclDataType type,void *&base,void *&body) {
            base=buffers.Allocate(Bytes(n+2*guard,4));body=static_cast<uint32_t *>(base)+guard;
            const std::vector<uint32_t> poison(n+2*guard,UINT32_MAX);
            Require(aclrtMemcpy(base,Bytes(poison.size(),4),poison.data(),Bytes(poison.size(),4),ACL_MEMCPY_HOST_TO_DEVICE),"poison merge arena");
            return Describe3D(tensors,q,n/(q*k),k,type,body);
        };
        void *base[2],*body[2];aclTensor *source[2];
        source[0]=allocate(q*count*k,ACL_FLOAT,base[0],body[0]);source[1]=allocate(q*count*k,ACL_INT32,base[1],body[1]);
        std::vector<float> values;std::vector<int32_t> ids;
        for(auto item:input){values.push_back(item.score);ids.push_back(item.id);}
        Require(aclrtMemcpy(body[0],Bytes(values.size(),4),values.data(),Bytes(values.size(),4),ACL_MEMCPY_HOST_TO_DEVICE),"merge input scores");
        Require(aclrtMemcpy(body[1],Bytes(ids.size(),4),ids.data(),Bytes(ids.size(),4),ACL_MEMCPY_HOST_TO_DEVICE),"merge input IDs");
        do {
            const uint64_t groups=(count+31)/32,n=q*groups*k;void *nextBase[2],*nextBody[2];aclTensor *next[2];
            next[0]=allocate(n,ACL_FLOAT,nextBase[0],nextBody[0]);next[1]=allocate(n,ACL_INT32,nextBase[1],nextBody[1]);
            uint64_t bytes=0;aclOpExecutor *executor=nullptr;
            Require(api.prepareMerge(source[0],source[1],k,leaves*k,next[0],next[1],&bytes,&executor),"prepare ragged global-ID merge");
            Check(executor!=nullptr,"No merge executor");Require(api.executeMerge(buffers.Allocate(bytes),bytes,executor,runtime.stream),"execute ragged merge");
            Require(aclrtSynchronizeStream(runtime.stream),"check merge sync");
            std::vector<Candidate> wanted(n);
            for(uint64_t query=0;query<q;++query)for(uint64_t group=0;group<groups;++group) {
                std::vector<Candidate> all;
                for(uint64_t child=group*32;child<std::min((group+1)*32,count);++child)
                    all.insert(all.end(),input.begin()+(query*count+child)*k,input.begin()+(query*count+child+1)*k);
                std::sort(all.begin(),all.end(),Better{});std::copy(all.begin(),all.begin()+k,wanted.begin()+(query*groups+group)*k);
            }
            std::vector<uint32_t> raw[2];
            for(size_t i=0;i<2;++i) {
                raw[i].resize(n+2*guard);Require(aclrtMemcpy(raw[i].data(),Bytes(raw[i].size(),4),nextBase[i],Bytes(raw[i].size(),4),ACL_MEMCPY_DEVICE_TO_HOST),"merge checks download");
                for(size_t j=0;j<guard;++j)Check(raw[i][j]==UINT32_MAX && raw[i][raw[i].size()-1-j]==UINT32_MAX,"Merge guard overwritten");
            }
            for(size_t i=0;i<n;++i) {uint32_t bits;std::memcpy(&bits,&wanted[i].score,4);
                Check(raw[0][guard+i]==bits && raw[1][guard+i]==static_cast<uint32_t>(wanted[i].id),"Exact hierarchical global-ID ranking mismatch");++checked;}
            input=std::move(wanted);source[0]=next[0];source[1]=next[1];count=groups;++calls;
        } while(count>1);
        Check(tensors.Close() && buffers.Close(),"Merge check cleanup failed");
    }
    Check(runtime.Close(),"Merge runtime cleanup failed");Check(fs::create_directories(output),"Cannot create merge check directory");
    std::ofstream record(output/"report.json");record<<"{\"status\":\"actual_global_id_merge_tree_boundaries_guards_and_bits_passed\",\"calls\":"<<calls
        <<",\"checked_candidates\":"<<checked<<",\"k_values\":[1,10,32],\"leaf_counts\":[1,31,32,33,1023,1024,1025]}\n";
    record.close();Check(bool(record),"Cannot save merge checks");
}
