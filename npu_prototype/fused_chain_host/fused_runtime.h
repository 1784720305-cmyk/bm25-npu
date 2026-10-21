#pragma once
// Independent resource snapshot of the accepted query-score runner, without its operator dependency.
// Fused API read from the actual installed A5 header supplied on 2026-10-21.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "acl/acl.h"
#include "aclnn_bm25_fused_score_top_k.h"
#include "aclnn_bm25_top_k_merge.h"
#include <cctype>
#include <numeric>
#include <sstream>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace fs = std::filesystem;
using Execute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using FusedPrepare = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, int64_t, int64_t, const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
using MergePrepare = aclnnStatus (*)(const aclTensor *, const aclTensor *, int64_t, int64_t,
    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
static_assert(std::is_same<decltype(&aclnnBm25FusedScoreTopKGetWorkspaceSize),FusedPrepare>::value,
    "Fused prepare differs from the actual header reviewed on 2026-10-21");
static_assert(std::is_same<decltype(&aclnnBm25FusedScoreTopK),Execute>::value);
static_assert(std::is_same<decltype(&aclnnBm25TopKMergeGetWorkspaceSize),MergePrepare>::value);
static_assert(std::is_same<decltype(&aclnnBm25TopKMerge),Execute>::value);

inline void Stage(const char *name) { std::cout << "stage=" << name << std::endl; }
inline void Check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
inline void Require(aclError code, const char *operation)
{
    if (code == ACL_ERROR_NONE) return;
    const char *detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(operation) + " failed, code=" + std::to_string(code) +
                             ", detail=" + (detail ? detail : "<none>"));
}
inline size_t Bytes(uint64_t count, size_t width)
{
    Check(count <= std::numeric_limits<size_t>::max()/width &&
          count <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())/width,
          "Array byte count exceeds host limits");
    return static_cast<size_t>(count)*width;
}
template<class T> std::vector<T> Read(const fs::path &path, bool allowEmpty = false)
{
    const auto size = fs::file_size(path);
    Check((allowEmpty || size > 0) && size % sizeof(T) == 0 && size/sizeof(T) <= INT32_MAX,
          "Invalid binary input size: " + path.string());
    const size_t bytes = Bytes(size/sizeof(T), sizeof(T));
    std::vector<T> values(bytes/sizeof(T));
    std::ifstream file(path,std::ios::binary);
    Check(file.is_open(), "Cannot open " + path.string());
    if (bytes) Check(static_cast<bool>(file.read(reinterpret_cast<char *>(values.data()),bytes)), "Cannot read " + path.string());
    return values;
}
template<class T> void Write(const fs::path &path, const std::vector<T> &values)
{
    std::ofstream file(path,std::ios::binary);
    const size_t bytes = Bytes(values.size(),sizeof(T));
    if (bytes) file.write(reinterpret_cast<const char *>(values.data()),bytes);
    file.close();
    Check(static_cast<bool>(file), "Cannot save " + path.string());
}

struct Runtime {
    bool initialized = false, selected = false;
    int device = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    bool Close()
    {
        bool ok = true;
        auto check = [&](aclError code, const char *name) {
            if (code != ACL_ERROR_NONE) { std::cerr << "cleanup_failed=" << name << " code=" << code << '\n'; ok = false; }
        };
        if (stream) {
            const auto code = aclrtSynchronizeStream(stream);
            if (code != ACL_ERROR_NONE) { check(code,"runtime synchronize"); return false; }
            check(aclrtDestroyStream(stream),"destroy stream"); stream = nullptr;
        }
        if (context) { check(aclrtDestroyContext(context),"destroy context"); context = nullptr; }
        if (selected) { check(aclrtResetDevice(device),"reset device"); selected = false; }
        if (initialized) { check(aclFinalize(),"finalize"); initialized = false; }
        return ok;
    }
    ~Runtime() { Close(); }
};

struct Tensors {
    aclrtStream stream = nullptr;
    std::vector<aclTensor *> items;
    aclTensor *Describe(int64_t rows, int64_t cols, aclDataType type, void *memory)
    {
        const int64_t dimensions[2] = {rows,cols}, strides[2] = {cols,1};
        auto *tensor = aclCreateTensor(dimensions,2,type,strides,0,ACL_FORMAT_ND,dimensions,2,memory);
        Check(tensor != nullptr,"aclCreateTensor 2D returned null"); items.push_back(tensor); return tensor;
    }
    aclTensor *Describe1D(int64_t count, aclDataType type, void *memory)
    {
        const int64_t stride = 1;
        auto *tensor = aclCreateTensor(&count,1,type,&stride,0,ACL_FORMAT_ND,&count,1,memory);
        Check(tensor != nullptr,"aclCreateTensor 1D returned null"); items.push_back(tensor); return tensor;
    }
    bool Close()
    {
        if (items.empty()) return true;
        if (stream && aclrtSynchronizeStream(stream) != ACL_ERROR_NONE) {
            std::cerr << "cleanup_failed=tensor synchronization; descriptors retained\n"; return false;
        }
        bool ok = true;
        for (auto *tensor : items) if (aclDestroyTensor(tensor) != ACL_ERROR_NONE) {
            std::cerr << "cleanup_failed=aclDestroyTensor\n"; ok = false;
        }
        items.clear(); return ok;
    }
    ~Tensors() { Close(); }
};
struct Buffers {
    aclrtStream stream = nullptr;
    std::vector<void *> items;
    void *Allocate(size_t bytes)
    {
        if (!bytes) return nullptr;
        void *p = nullptr;
        Require(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_NORMAL_ONLY),"aclrtMalloc");
        Check(p != nullptr,"allocator returned null"); items.push_back(p); return p;
    }
    bool Close()
    {
        if (items.empty()) return true;
        if (stream && aclrtSynchronizeStream(stream) != ACL_ERROR_NONE) {
            std::cerr << "cleanup_failed=buffer synchronization; memory retained\n"; return false;
        }
        bool ok = true;
        for (auto *p : items) if (aclrtFree(p) != ACL_ERROR_NONE) {
            std::cerr << "cleanup_failed=aclrtFree\n"; ok = false;
        }
        items.clear(); return ok;
    }
    ~Buffers() { Close(); }
};

class ApiLibrary {
    fs::path path;
    void *handle = nullptr;
public:
    explicit ApiLibrary(const fs::path &vendor) : path(fs::canonical(vendor/"op_api/lib/libcust_opapi.so"))
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
    ApiLibrary(const ApiLibrary &) = delete;
    ApiLibrary &operator=(const ApiLibrary &) = delete;
    // Keep handles to process exit because CANN registries may keep callbacks.
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
struct Alternate {
    std::vector<float> partialScores, topScores, groupScores;
    std::vector<int32_t> partialIds, topIds, groupIds;
};
struct Request {
    std::string label;
    fs::path directory;
    std::vector<int64_t> offsets;
    std::vector<int32_t> terms;
    std::vector<float> expected, actual, partialScores, topScores, groupScores;
    std::vector<int32_t> partialIds, topIds, groupIds;
    Alternate disabled;
    uint32_t disabledPlaceholder = UINT32_MAX;
    uint64_t workspace[2][3] = {};
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
            Check(0 <= request.offsets[i-1] && request.offsets[i-1] <= request.offsets[i] && request.offsets[i]-request.offsets[i-1] <= 64,"Invalid query boundary or more than 64 words in one query");
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
        if (shards == 64) for (size_t group = 0; group < 2; ++group)
            checkRange(static_cast<size_t>(q)*docs+group*(docs/2),docs/2,r.groupIds,r.groupScores,
                       (static_cast<size_t>(q)*2+group)*k);
        checkRange(static_cast<size_t>(q)*docs,docs,r.topIds,r.topScores,static_cast<size_t>(q)*k);
    }
}
