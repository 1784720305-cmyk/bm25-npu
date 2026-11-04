#ifndef BM25_ASCENDHOST_SRC_BM25_FUSED_RUNTIME_H
#define BM25_ASCENDHOST_SRC_BM25_FUSED_RUNTIME_H
#include "ascendhost/src/bm25/runtime.h"
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "aclnn_bm25_fused_score_top_k.h"
#include "aclnn_bm25_top_k_merge.h"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/bm25/index_data.h"
#include "common/bm25/error.h"

namespace ascend::bm25 {
namespace fs = std::filesystem;
using Execute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using FusedPrepare = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
                                     const aclTensor *, const aclTensor *, const aclTensor *, int64_t, int64_t, int64_t,
                                     const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *,
                                     aclOpExecutor **);
using MergePrepare = aclnnStatus (*)(const aclTensor *, const aclTensor *, int64_t, int64_t, const aclTensor *,
                                     const aclTensor *, uint64_t *, aclOpExecutor **);
static_assert(std::is_same<decltype(&aclnnBm25FusedScoreTopKGetWorkspaceSize), FusedPrepare>::value,
              "Fused prepare differs from the actual header reviewed on 2026-10-21");
static_assert(std::is_same<decltype(&aclnnBm25FusedScoreTopK), Execute>::value);
static_assert(std::is_same<decltype(&aclnnBm25TopKMergeGetWorkspaceSize), MergePrepare>::value);
static_assert(std::is_same<decltype(&aclnnBm25TopKMerge), Execute>::value);

inline void Require(aclError code, const char *operation) { RequireAcl(code, operation); }

struct Tensors {
    Tensors() = default;
    Tensors(const Tensors &) = delete;
    Tensors &operator=(const Tensors &) = delete;
    Tensors(Tensors &&) = delete;
    Tensors &operator=(Tensors &&) = delete;
    aclrtStream stream = nullptr;
    std::vector<aclTensor *> items;
    aclTensor *Describe(int64_t rows, int64_t cols, aclDataType type, void *memory)
    {
        const int64_t dimensions[2] = {rows, cols};
        const int64_t strides[2] = {cols, 1};
        auto *tensor = aclCreateTensor(dimensions, 2, type, strides, 0, ACL_FORMAT_ND, dimensions, 2, memory);
        Check(tensor != nullptr, "aclCreateTensor 2D returned null");
        items.push_back(tensor);
        return tensor;
    }
    aclTensor *Describe1D(int64_t count, aclDataType type, void *memory)
    {
        const int64_t stride = 1;
        auto *tensor = aclCreateTensor(&count, 1, type, &stride, 0, ACL_FORMAT_ND, &count, 1, memory);
        Check(tensor != nullptr, "aclCreateTensor 1D returned null");
        items.push_back(tensor);
        return tensor;
    }
    bool Close();
    ~Tensors();
};
struct Buffers {
    Buffers() = default;
    Buffers(const Buffers &) = delete;
    Buffers &operator=(const Buffers &) = delete;
    Buffers(Buffers &&) = delete;
    Buffers &operator=(Buffers &&) = delete;
    aclrtStream stream = nullptr;
    std::vector<void *> items;
    void *Allocate(size_t bytes)
    {
        if (!bytes)
            return nullptr;
        void *p = nullptr;
        Require(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
        Check(p != nullptr, "allocator returned null");
        items.push_back(p);
        return p;
    }
    bool Close();
    ~Buffers();
};

class ApiLibrary {
    fs::path path;
    void *handle = nullptr;

public:
    explicit ApiLibrary(const fs::path &vendor) : path(fs::canonical(vendor / "op_api/lib/libcust_opapi.so"))
    {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            const char *e = dlerror();
            throw std::runtime_error("dlopen failed: " + path.string() + ": " + (e ? e : "unknown"));
        }
        std::cout << "loaded_api_library=" << path << std::endl;
    }
    template <class Function> Function Symbol(const char *name)
    {
        dlerror();
        void *address = dlsym(handle, name);
        const char *error = dlerror();
        Check(!error && address, std::string("dlsym failed: ") + name + ": " + (error ? error : "null"));
        Dl_info owner{};
        Check(dladdr(address, &owner) && owner.dli_fname && fs::equivalent(path, owner.dli_fname),
              std::string("Unexpected API symbol owner: ") + name);
        static_assert(sizeof(Function) == sizeof(address));
        Function result;
        std::memcpy(&result, &address, sizeof(result));
        std::cout << "verified_api_symbol=" << name << " owner=" << owner.dli_fname << std::endl;
        return result;
    }
    ~ApiLibrary() = default;
    ApiLibrary(ApiLibrary &&) = delete;
    ApiLibrary &operator=(ApiLibrary &&) = delete;
    ApiLibrary(const ApiLibrary &) = delete;
    ApiLibrary &operator=(const ApiLibrary &) = delete;
    // Keep handles to process exit because CANN registries may keep callbacks.
};
inline aclTensor *Describe3D(Tensors &t, int64_t q, int64_t s, int64_t k, aclDataType type, void *memory)
{
    const int64_t dims[3] = {q, s, k};
    const int64_t strides[3] = {s * k, k, 1};
    auto *tensor = aclCreateTensor(dims, 3, type, strides, 0, ACL_FORMAT_ND, dims, 3, memory);
    Check(tensor != nullptr, "aclCreateTensor 3D returned null");
    t.items.push_back(tensor);
    return tensor;
}
template <class T> bool Same(const std::vector<T> &a, const std::vector<T> &b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), Bytes(a.size(), sizeof(T))) == 0;
}

} // namespace ascend::bm25
#endif // BM25_ASCENDHOST_SRC_BM25_FUSED_RUNTIME_H
