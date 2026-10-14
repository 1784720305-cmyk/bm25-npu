#pragma once
// Same-process, same-stream device pipeline using confirmed generated ACLNN signatures.
// No CPU contribution/accumulation/ranking fallback and no ATC/OM.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "acl/acl.h"
#include "aclnn_bm25_top_k.h"
#include "aclnn_bm25_contribution.h"
#include "aclnn_bm25_accumulate.h"
#include <dlfcn.h>
#include <type_traits>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using PrepareSignature = aclnnStatus (*)(const aclTensor *, int64_t, const aclTensor *, const aclTensor *,
                                        uint64_t *, aclOpExecutor **);
using ExecuteSignature = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
static_assert(std::is_same<decltype(&aclnnBm25TopKGetWorkspaceSize), PrepareSignature>::value,
              "Generated prepare signature differs from the A5 header provided");
static_assert(std::is_same<decltype(&aclnnBm25TopK), ExecuteSignature>::value,
              "Generated execute signature differs from the A5 header provided");

void Stage(const char *name) { std::cout << "stage=" << name << std::endl; }

void Require(aclError error, const char *operation)
{
    if (error == ACL_ERROR_NONE) return;
    const char *detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(operation) + " failed, code=" + std::to_string(error) +
                             ", detail=" + (detail == nullptr ? "<none>" : detail));
}

template<class T> std::vector<T> Read(const fs::path &path)
{
    const auto bytes = fs::file_size(path);
    if (bytes == 0 || bytes % sizeof(T) != 0 ||
        bytes > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()) ||
        bytes > std::numeric_limits<size_t>::max() ||
        bytes / sizeof(T) > static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()))
        throw std::runtime_error("Invalid binary input size: " + path.string());
    std::vector<T> result(static_cast<size_t>(bytes / sizeof(T)));
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("Cannot read: " + path.string());
    return result;
}

// Same runtime/resource ownership as the device-verified contribution diagnostic.
struct Runtime {
    bool initialized = false;
    bool deviceSelected = false;
    int device = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    bool Close()
    {
        bool ok = true;
        auto check = [&](aclError code, const char *name) {
            if (code != ACL_ERROR_NONE) { std::cerr << "cleanup_failed=" << name << " code=" << code << '\n'; ok = false; }
        };
        if (stream != nullptr) {
            const auto status = aclrtSynchronizeStream(stream);
            if (status != ACL_ERROR_NONE) {
                check(status, "runtime synchronize");
                return false; // Do not release resources while device work may be pending.
            }
            check(aclrtDestroyStream(stream), "aclrtDestroyStream"); stream = nullptr;
        }
        if (context != nullptr) { check(aclrtDestroyContext(context), "aclrtDestroyContext"); context = nullptr; }
        if (deviceSelected) { check(aclrtResetDevice(device), "aclrtResetDevice"); deviceSelected = false; }
        if (initialized) { check(aclFinalize(), "aclFinalize"); initialized = false; }
        return ok;
    }
    ~Runtime() { Close(); }
};

struct Resources {
    aclrtStream stream = nullptr;
    std::vector<void *> memory;
    std::vector<aclTensor *> tensors;
    bool Close()
    {
        if (stream != nullptr && aclrtSynchronizeStream(stream) != ACL_ERROR_NONE) {
            std::cerr << "cleanup_failed=resource synchronization; device buffers not freed while work may be pending\n";
            return false;
        }
        bool ok = true;
        for (auto *p : tensors) {
            if (aclDestroyTensor(p) != 0) { std::cerr << "cleanup_failed=aclDestroyTensor\n"; ok = false; }
        }
        tensors.clear();
        for (auto *p : memory) {
            if (aclrtFree(p) != ACL_ERROR_NONE) { std::cerr << "cleanup_failed=aclrtFree\n"; ok = false; }
        }
        memory.clear();
        stream = nullptr;
        return ok;
    }
    ~Resources() { Close(); }
    void *Allocate(size_t bytes)
    {
        if (bytes == 0) return nullptr;
        void *p = nullptr;
        Require(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
        memory.push_back(p);
        return p;
    }
    aclTensor *Describe1D(int64_t count, aclDataType type, void *deviceData)
    {
        const int64_t stride = 1;
        auto *tensor = aclCreateTensor(&count, 1, type, &stride, 0, ACL_FORMAT_ND, &count, 1, deviceData);
        if (tensor == nullptr) throw std::runtime_error("aclCreateTensor returned null");
        tensors.push_back(tensor);
        return tensor;
    }
    aclTensor *Describe(int64_t rows, int64_t cols, aclDataType type, void *deviceData)
    {
        const int64_t dimensions[2] = {rows, cols};
        const int64_t strides[2] = {cols, 1};
        auto *tensor = aclCreateTensor(dimensions, 2, type, strides, 0, ACL_FORMAT_ND,
                                      dimensions, 2, deviceData);
        if (tensor == nullptr) throw std::runtime_error("aclCreateTensor returned null");
        tensors.push_back(tensor);
        return tensor;
    }
};


int64_t ParseInteger(const char *text, const char *name, int64_t minimum, int64_t maximum)
{
    size_t used = 0;
    const int64_t value = std::stoll(text, &used);
    if (used != std::string(text).size() || value < minimum || value > maximum)
        throw std::runtime_error(std::string("Invalid ") + name);
    return value;
}

size_t CheckedBytes(uint64_t count, size_t elementSize)
{
    if (count > std::numeric_limits<size_t>::max() / elementSize ||
        count > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / elementSize)
        throw std::runtime_error("Tensor byte count exceeds host limits");
    return static_cast<size_t>(count) * elementSize;
}

void ValidateOutput(const std::vector<float> &input, const std::vector<float> &scores,
                    const std::vector<int32_t> &ids, int64_t queries, int64_t docs, int64_t k)
{
    // Only validate returned records; independent full ranking is done by the separate checker.
    for (int64_t query = 0; query < queries; ++query) {
        const size_t outputBase = static_cast<size_t>(query * k);
        const size_t inputBase = static_cast<size_t>(query * docs);
        for (int64_t rank = 0; rank < k; ++rank) {
            const size_t index = outputBase + static_cast<size_t>(rank);
            const int32_t id = ids[index];
            if (id < 0 || id >= docs || !std::isfinite(scores[index]) || scores[index] < 0)
                throw std::runtime_error("Invalid device Top-K value at record " + std::to_string(index));
            if (std::memcmp(&scores[index], &input[inputBase + static_cast<size_t>(id)], sizeof(float)) != 0)
                throw std::runtime_error("Selected score differs from input document score");
            for (int64_t earlier = 0; earlier < rank; ++earlier) {
                if (ids[outputBase + static_cast<size_t>(earlier)] == id)
                    throw std::runtime_error("Duplicate document ID within a query");
            }
            if (rank > 0 && (scores[index] > scores[index - 1] ||
                (scores[index] == scores[index - 1] && id < ids[index - 1])))
                throw std::runtime_error("Device Top-K order violates score/ID rules");
        }
    }
}

template<class T> void Write(const fs::path &path, const std::vector<T> &values)
{
    const size_t bytes = CheckedBytes(values.size(), sizeof(T));
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(bytes));
    output.close();
    if (!output) throw std::runtime_error("Cannot save: " + path.string());
}


using ContributionPrepare = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *,
                                            const aclTensor *, uint64_t *, aclOpExecutor **);
using AccumulatePrepare = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *,
                                        uint64_t *, aclOpExecutor **);
static_assert(std::is_same<decltype(&aclnnBm25ContributionGetWorkspaceSize), ContributionPrepare>::value);
static_assert(std::is_same<decltype(&aclnnBm25AccumulateGetWorkspaceSize), AccumulatePrepare>::value);
static_assert(std::is_same<decltype(&aclnnBm25Contribution), ExecuteSignature>::value);
static_assert(std::is_same<decltype(&aclnnBm25Accumulate), ExecuteSignature>::value);

class Library {
    fs::path path;
    void *handle = nullptr;
public:
    explicit Library(const fs::path &filename) : path(fs::canonical(filename))
    {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            const char *error = dlerror();
            throw std::runtime_error("dlopen failed: " + path.string() + ": " + (error ? error : "<none>"));
        }
        std::cout << "loaded_api_library=" << path << std::endl;
    }
    Library(const Library &) = delete;
    Library &operator=(const Library &) = delete;
    // Keep code resident until process exit: CANN registries may retain API-library callbacks.
    // Device buffers/tensors/stream/context are explicitly released before successful return.
    template<class Function> Function Symbol(const char *name)
    {
        dlerror();
        void *symbol = dlsym(handle, name);
        const char *error = dlerror();
        if (error != nullptr || symbol == nullptr)
            throw std::runtime_error(std::string("dlsym failed: ") + name + ": " + (error ? error : "null"));
        Dl_info owner{};
        if (dladdr(symbol, &owner) == 0 || owner.dli_fname == nullptr || !fs::equivalent(path, owner.dli_fname))
            throw std::runtime_error(std::string("API symbol resolved from an unexpected library: ") + name);
        std::cout << "verified_api_symbol=" << name << " owner=" << owner.dli_fname << std::endl;
        static_assert(sizeof(Function) == sizeof(symbol), "Linux function-pointer representation required");
        Function function;
        std::memcpy(&function, &symbol, sizeof(function)); // POSIX dlsym function address.
        return function;
    }
};

struct Interfaces {
    Library contribution;
    Library accumulate;
    Library topk;
    ContributionPrepare prepareContribution;
    AccumulatePrepare prepareAccumulate;
    PrepareSignature prepareTopK;
    ExecuteSignature executeContribution, executeAccumulate, executeTopK;
    explicit Interfaces(const fs::path &vendors)
        : contribution(vendors / "bm25/op_api/lib/libcust_opapi.so"),
          accumulate(vendors / "bm25_accumulate/op_api/lib/libcust_opapi.so"),
          topk(vendors / "bm25_topk/op_api/lib/libcust_opapi.so")
    {
        prepareContribution = contribution.Symbol<ContributionPrepare>("aclnnBm25ContributionGetWorkspaceSize");
        executeContribution = contribution.Symbol<ExecuteSignature>("aclnnBm25Contribution");
        prepareAccumulate = accumulate.Symbol<AccumulatePrepare>("aclnnBm25AccumulateGetWorkspaceSize");
        executeAccumulate = accumulate.Symbol<ExecuteSignature>("aclnnBm25Accumulate");
        prepareTopK = topk.Symbol<PrepareSignature>("aclnnBm25TopKGetWorkspaceSize");
        executeTopK = topk.Symbol<ExecuteSignature>("aclnnBm25TopK");
    }
};

