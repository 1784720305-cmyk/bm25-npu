#pragma once
// Frozen resource and query-score API implementation from the device-verified score diagnostic.
#include "../query_score_host/query_score_runtime.h"
#include "aclnn_bm25_top_k.h"
#include "aclnn_bm25_top_k_merge.h"
#include <cctype>
#include <numeric>
#include <sstream>

// Both generated selection signatures were supplied from the actual installed A5 headers
// and are already used by the device-verified sharded runner. Recheck at compile time.
using TopPrepare = aclnnStatus (*)(const aclTensor *, int64_t, const aclTensor *, const aclTensor *,
                                  uint64_t *, aclOpExecutor **);
using MergePrepare = aclnnStatus (*)(const aclTensor *, const aclTensor *, int64_t, int64_t,
                                    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
static_assert(std::is_same<decltype(&aclnnBm25TopKGetWorkspaceSize),TopPrepare>::value);
static_assert(std::is_same<decltype(&aclnnBm25TopK),QueryExecute>::value);
static_assert(std::is_same<decltype(&aclnnBm25TopKMergeGetWorkspaceSize),MergePrepare>::value);
static_assert(std::is_same<decltype(&aclnnBm25TopKMerge),QueryExecute>::value);

class SelectionLibrary {
    fs::path path;
    void *handle = nullptr;
public:
    explicit SelectionLibrary(const fs::path &vendor) : path(fs::canonical(vendor/"op_api/lib/libcust_opapi.so"))
    {
        handle = dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
        if (!handle) { const char *e = dlerror(); throw std::runtime_error("dlopen failed: "+path.string()+": "+(e?e:"unknown")); }
        std::cout << "loaded_api_library=" << path << std::endl;
    }
    template<class Function> Function Symbol(const char *name)
    {
        dlerror(); void *address = dlsym(handle,name); const char *error = dlerror();
        Check(!error && address,std::string("dlsym failed: ")+name+": "+(error?error:"null"));
        Dl_info owner{};
        Check(dladdr(address,&owner) && owner.dli_fname && fs::equivalent(path,owner.dli_fname),
              std::string("Unexpected API symbol owner: ")+name);
        static_assert(sizeof(Function) == sizeof(address));
        Function result; std::memcpy(&result,&address,sizeof(result));
        std::cout << "verified_api_symbol=" << name << " owner=" << owner.dli_fname << std::endl;
        return result;
    }
    SelectionLibrary(const SelectionLibrary &) = delete;
    SelectionLibrary &operator=(const SelectionLibrary &) = delete;
    // Keep handles to process exit because CANN registries may keep callbacks.
};
struct ChainApi {
    QueryApi score;
    SelectionLibrary top, merge;
    TopPrepare prepareTop;
    QueryExecute executeTop, executeMerge;
    MergePrepare prepareMerge;
    explicit ChainApi(const fs::path &vendors)
        : score(vendors/"bm25_query_score"), top(vendors/"bm25_topk_radix_collect"), merge(vendors/"bm25_topk_merge")
    {
        prepareTop = top.Symbol<TopPrepare>("aclnnBm25TopKGetWorkspaceSize");
        executeTop = top.Symbol<QueryExecute>("aclnnBm25TopK");
        prepareMerge = merge.Symbol<MergePrepare>("aclnnBm25TopKMergeGetWorkspaceSize");
        executeMerge = merge.Symbol<QueryExecute>("aclnnBm25TopKMerge");
    }
};

inline int64_t Number(const char *text,int64_t low,int64_t high)
{
    size_t parsed = 0; const int64_t value = std::stoll(text,&parsed);
    Check(parsed == std::strlen(text) && value >= low && value <= high,"Invalid integer argument");
    return value;
}
inline aclTensor *Describe3D(Tensors &t,int64_t q,int64_t s,int64_t k,aclDataType type,void *memory)
{
    const int64_t dims[3] = {q,s,k}, strides[3] = {s*k,k,1};
    auto *tensor = aclCreateTensor(dims,3,type,strides,0,ACL_FORMAT_ND,dims,3,memory);
    Check(tensor != nullptr,"aclCreateTensor 3D returned null"); t.items.push_back(tensor); return tensor;
}
template<class T> bool Same(const std::vector<T> &a,const std::vector<T> &b)
{
    return a.size() == b.size() && std::memcmp(a.data(),b.data(),Bytes(a.size(),sizeof(T))) == 0;
}
struct Index {
    std::vector<int64_t> offsets;
    std::vector<int32_t> docs;
    std::vector<float> tf, norm, idf;
    explicit Index(const fs::path &directory)
        : offsets(Read<int64_t>(directory/"term_offsets.bin")), docs(Read<int32_t>(directory/"posting_doc_ids.bin")),
          tf(Read<float>(directory/"posting_tf.bin")), norm(Read<float>(directory/"doc_norm.bin")),
          idf(Read<float>(directory/"idf.bin"))
    {
        Check(offsets.size() == idf.size()+1 && docs.size() == tf.size() &&
              offsets.front() == 0 && offsets.back() == static_cast<int64_t>(docs.size()), "Index dimensions/endpoints differ");
        for (float value : norm) Check(std::isfinite(value) && value > 0,"Invalid article norm");
        for (size_t term = 0; term < idf.size(); ++term) {
            Check(std::isfinite(idf[term]) && idf[term] >= 0,"Invalid IDF");
            const int64_t begin = offsets[term], end = offsets[term+1];
            Check(begin >= 0 && begin <= end && end <= static_cast<int64_t>(docs.size()),"Invalid posting bounds");
            for (int64_t pos = begin; pos < end; ++pos) {
                Check(docs[pos] >= 0 && static_cast<size_t>(docs[pos]) < norm.size() &&
                      (pos == begin || docs[pos-1] < docs[pos]),"Unsorted/duplicate/out-of-range posting ID");
                Check(std::isfinite(tf[pos]) && tf[pos] >= 1 && tf[pos] <= 16777216.0F &&
                      std::floor(tf[pos]) == tf[pos],"Invalid posting TF");
            }
        }
    }
};
struct Request {
    std::string label;
    fs::path directory;
    std::vector<int64_t> offsets;
    std::vector<int32_t> terms;
    std::vector<float> expected, actual, partialScores, topScores;
    std::vector<int32_t> partialIds, topIds;
    uint64_t workspace[3] = {0,0,0};
    bool repeated = false;
    int64_t Queries() const { return static_cast<int64_t>(offsets.size()-1); }
};

std::vector<Request> LoadRequests(const fs::path &plan, size_t docs)
{
    std::ifstream file(plan);
    Check(file.is_open(),"Cannot read request plan");
    std::vector<Request> result;
    std::map<std::string,bool> labels;
    std::string line;
    uint64_t totalScores = 0;
    while (std::getline(file,line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto separator = line.find('\t');
        Check(separator != std::string::npos && separator > 0 && line.find('\t',separator+1) == std::string::npos,
              "Request plan needs label<TAB>query-directory");
        Request request;
        request.label = line.substr(0,separator);
        Check(std::all_of(request.label.begin(),request.label.end(),[](unsigned char c){return std::isalnum(c) || c == '_';}) &&
              labels.emplace(request.label,true).second,"Unsafe/duplicate request label");
        request.directory = fs::canonical(line.substr(separator+1));
        request.offsets = Read<int64_t>(request.directory/"query_offsets.bin");
        request.terms = Read<int32_t>(request.directory/"query_term_ids.bin",true);
        Check(request.offsets.size() >= 2 && request.offsets.front() == 0 &&
              request.offsets.back() == static_cast<int64_t>(request.terms.size()),"Bad query boundary endpoints");
        for (size_t i = 1; i < request.offsets.size(); ++i)
            Check(0 <= request.offsets[i-1] && request.offsets[i-1] <= request.offsets[i],"Nonmonotonic query boundaries");
        for (int32_t word : request.terms) Check(word >= 0 && word < INT32_MAX,"Bad query ID");
        Check(static_cast<uint64_t>(request.Queries()) <= static_cast<uint64_t>(INT32_MAX)/docs,"Q*N exceeds operator limit");
        const uint64_t count = static_cast<uint64_t>(request.Queries())*docs;
        Check(count <= 16777216 && totalScores <= 16777216-count,"Diagnostic total score budget exceeded");
        totalScores += count;
        request.expected = Read<float>(request.directory/"expected_scores.bin");
        Check(request.expected.size() == count,"Reference dimensions differ");
        for (float value : request.expected) Check(std::isfinite(value) && value >= 0,"Invalid reference value");
        result.push_back(std::move(request));
        Check(result.size() <= 32,"Diagnostic plan exceeds 32 requests");
    }
    Check(file.eof() && !result.empty(),"Empty/unreadable request plan");
    return result;
}

void CompareReference(const Request &request)
{
    Check(request.actual.size() == request.expected.size(),"Output dimensions differ");
    for (size_t i = 0; i < request.actual.size(); ++i) {
        const double a = request.actual[i], e = request.expected[i];
        if (!std::isfinite(a) || a < 0 || std::abs(a-e) > 1e-6+2e-6*std::abs(e)) {
            std::ostringstream message;
            message.precision(17);
            message << "Device/reference mismatch: " << request.label << " element=" << i
                    << " actual=" << a << " expected=" << e << " tolerance=" << 1e-6+2e-6*std::abs(e);
            throw std::runtime_error(message.str());
        }
    }
}


inline void CheckRanking(const Request &r,size_t docs,int64_t k,int64_t shards)
{
    const size_t width = docs/static_cast<size_t>(shards);
    auto checkRange = [&](size_t base,size_t count,const std::vector<int32_t> &ids,
                          const std::vector<float> &scores,size_t out) {
        std::vector<int32_t> order(count); std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](int32_t a,int32_t b) {
            const float x = r.actual[base+a], y = r.actual[base+b];
            return x != y ? x > y : a < b;
        });
        for (int64_t rank = 0; rank < k; ++rank) {
            const size_t pos = out+static_cast<size_t>(rank);
            Check(ids.at(pos) == order[rank],"Device Top-K ID/tie rule mismatch");
            const float expected = r.actual[base+static_cast<size_t>(order[rank])];
            Check(std::memcmp(&scores.at(pos),&expected,4) == 0,"Device Top-K score bits mismatch");
        }
    };
    for (int64_t q = 0; q < r.Queries(); ++q) {
        for (int64_t s = 0; s < shards; ++s)
            checkRange(static_cast<size_t>(q)*docs+s*width,width,r.partialIds,r.partialScores,
                       static_cast<size_t>(q*shards+s)*k);
        checkRange(static_cast<size_t>(q)*docs,docs,r.topIds,r.topScores,static_cast<size_t>(q)*k);
    }
}
