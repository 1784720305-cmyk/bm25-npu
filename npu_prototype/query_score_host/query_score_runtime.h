#pragma once
// Resource/API pattern follows existing device-verified runners; no legacy operator dependencies.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "acl/acl.h"
#include "aclnn_bm25_query_score.h"
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
// Confirmed from the installed Ascend950 header supplied on 2026-10-20.
using QueryPrepare = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
using QueryExecute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
static_assert(std::is_same<decltype(&aclnnBm25QueryScoreGetWorkspaceSize), QueryPrepare>::value,
              "Installed prepare signature differs from the header actually reviewed");
static_assert(std::is_same<decltype(&aclnnBm25QueryScore), QueryExecute>::value,
              "Installed execute signature differs from the header actually reviewed");

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

class QueryApi {
    fs::path path;
    void *handle = nullptr;
    template<class Function> Function Symbol(const char *name)
    {
        dlerror();
        void *symbol = dlsym(handle,name);
        const char *error = dlerror();
        Check(error == nullptr && symbol != nullptr,std::string("dlsym failed: ")+name+": "+(error ? error : "null"));
        Dl_info owner{};
        Check(dladdr(symbol,&owner) != 0 && owner.dli_fname && fs::equivalent(path,owner.dli_fname),
              std::string("Unexpected API symbol owner: ")+name);
        static_assert(sizeof(Function) == sizeof(symbol));
        Function function; std::memcpy(&function,&symbol,sizeof(function));
        std::cout << "verified_api_symbol=" << name << " owner=" << owner.dli_fname << std::endl;
        return function;
    }
public:
    QueryPrepare prepare = nullptr;
    QueryExecute execute = nullptr;
    explicit QueryApi(const fs::path &vendor) : path(fs::canonical(vendor/"op_api/lib/libcust_opapi.so"))
    {
        handle = dlopen(path.c_str(),RTLD_NOW | RTLD_LOCAL);
        if (!handle) { const char *error = dlerror(); throw std::runtime_error("dlopen failed: "+path.string()+": "+(error ? error : "unknown")); }
        std::cout << "loaded_api_library=" << path << std::endl;
        prepare = Symbol<QueryPrepare>("aclnnBm25QueryScoreGetWorkspaceSize");
        execute = Symbol<QueryExecute>("aclnnBm25QueryScore");
    }
    QueryApi(const QueryApi &) = delete;
    QueryApi &operator=(const QueryApi &) = delete;
    // Keep library resident to process exit: CANN registries may retain its callbacks.
};
