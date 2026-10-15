// Merge signatures confirmed from the installed A5 header supplied on 2026-10-15.
// Unit diagnostic: reads candidate arrays only; never creates a CPU output substitute.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "acl/acl.h"
#include "aclnn_bm25_top_k_merge.h"
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
using PrepareSignature = aclnnStatus (*)(const aclTensor *, const aclTensor *, int64_t, int64_t, const aclTensor *, const aclTensor *,
                                        uint64_t *, aclOpExecutor **);
using ExecuteSignature = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
static_assert(std::is_same<decltype(&aclnnBm25TopKMergeGetWorkspaceSize), PrepareSignature>::value,
              "Generated prepare signature differs from the A5 header provided");
static_assert(std::is_same<decltype(&aclnnBm25TopKMerge), ExecuteSignature>::value,
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

template<class T> void Write(const fs::path &path, const std::vector<T> &values)
{
    const size_t bytes = CheckedBytes(values.size(), sizeof(T));
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(bytes));
    output.close();
    if (!output) throw std::runtime_error("Cannot save: " + path.string());
}

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


aclTensor *DescribeCandidates(Resources &resources, int64_t q, int64_t shards, int64_t k,
                              aclDataType type, void *memory)
{
    const int64_t shape[3] = {q, shards, k};
    const int64_t strides[3] = {shards*k, k, 1};
    auto *tensor = aclCreateTensor(shape, 3, type, strides, 0, ACL_FORMAT_ND, shape, 3, memory);
    if (!tensor) throw std::runtime_error("Cannot describe candidate tensor");
    resources.tensors.push_back(tensor);
    return tensor;
}

void ValidateCandidates(const std::vector<float> &scores, const std::vector<int32_t> &ids,
                        int64_t queries, int64_t shards, int64_t docs, int64_t k)
{
    if (scores.size() != static_cast<uint64_t>(queries)*shards*k || ids.size() != scores.size())
        throw std::runtime_error("Candidate file sizes differ from dimensions");
    for (int64_t row = 0; row < queries*shards; ++row) {
        const size_t base = static_cast<size_t>(row*k);
        for (int64_t rank = 0; rank < k; ++rank) {
            const size_t i = base + static_cast<size_t>(rank);
            if (!std::isfinite(scores[i]) || scores[i] < 0 || ids[i] < 0 || ids[i] >= docs/shards)
                throw std::runtime_error("Invalid partial score or block-local ID");
            if (rank && (scores[i] > scores[i-1] || (scores[i] == scores[i-1] && ids[i] < ids[i-1])))
                throw std::runtime_error("Candidate list is not sorted by score/local ID");
            for (int64_t earlier = 0; earlier < rank; ++earlier)
                if (ids[base+static_cast<size_t>(earlier)] == ids[i])
                    throw std::runtime_error("Duplicate block-local candidate ID");
        }
    }
}

void ValidateSelected(const std::vector<float> &partial, const std::vector<int32_t> &local,
                      const std::vector<float> &scores, const std::vector<int32_t> &ids,
                      int64_t queries, int64_t shards, int64_t docs, int64_t k)
{
    // Membership/order checks only; full independent ranking is the Python checker's job.
    const int64_t width = docs/shards;
    for (int64_t q = 0; q < queries; ++q) {
        for (int64_t rank = 0; rank < k; ++rank) {
            const size_t i = static_cast<size_t>(q*k+rank);
            if (!std::isfinite(scores[i]) || scores[i] < 0 || ids[i] < 0 || ids[i] >= docs)
                throw std::runtime_error("Invalid actual merge output");
            if (rank && (scores[i] > scores[i-1] || (scores[i] == scores[i-1] && ids[i] < ids[i-1])))
                throw std::runtime_error("Actual merge ranking violates score/global ID rule");
            for (int64_t earlier = 0; earlier < rank; ++earlier)
                if (ids[static_cast<size_t>(q*k+earlier)] == ids[i])
                    throw std::runtime_error("Actual merge contains duplicate document IDs");
            const size_t start = static_cast<size_t>((q*shards + ids[i]/width)*k);
            bool found = false;
            for (int64_t r = 0; r < k; ++r) {
                const size_t j = start + static_cast<size_t>(r);
                if (local[j] == ids[i]%width && std::memcmp(&scores[i], &partial[j], sizeof(float)) == 0)
                    found = true;
            }
            if (!found) throw std::runtime_error("Actual output is not an unchanged input candidate");
        }
    }
}

int main(int argc, char **argv)
{
    try {
        if (argc != 9) {
            std::cerr << "Usage: bm25_topk_merge_runner INPUT OUTPUT Q DOCS SHARDS K DEVICE API_LIBRARY\n";
            return 2;
        }
        static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4);
        static_assert(std::numeric_limits<float>::is_iec559);
        const uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian) != 1) throw std::runtime_error("Little endian required");
        Stage("validate_inputs");
        const int64_t q = ParseInteger(argv[3], "queries", 1, INT32_MAX);
        const int64_t docs = ParseInteger(argv[4], "docs", 1, INT32_MAX);
        const int64_t shards = ParseInteger(argv[5], "shards", 1, 32);
        const int64_t k = ParseInteger(argv[6], "K", 1, 32);
        const int64_t device = ParseInteger(argv[7], "device", 0, INT32_MAX);
        if (q*docs > INT32_MAX || docs%shards != 0 || docs/shards < k)
            throw std::runtime_error("Unsupported dimensions, nondivisible documents or too few documents per shard");
        const fs::path input = fs::absolute(argv[1]), output = fs::absolute(argv[2]);
        if (fs::exists(output)) throw std::runtime_error("Choose fresh output directory");
        const auto partial = Read<float>(input / "partial_scores.bin");
        const auto local = Read<int32_t>(input / "partial_ids.bin");
        ValidateCandidates(partial, local, q, shards, docs, k);
        const size_t inputBytes = CheckedBytes(partial.size(), 4);
        const size_t outputCount = static_cast<size_t>(q*k);
        const size_t outputBytes = CheckedBytes(outputCount, 4);
        Library library(fs::absolute(argv[8]));
        const auto prepare = library.Symbol<PrepareSignature>("aclnnBm25TopKMergeGetWorkspaceSize");
        const auto execute = library.Symbol<ExecuteSignature>("aclnnBm25TopKMerge");
        Runtime runtime;
        runtime.device = static_cast<int>(device);
        Stage("runtime_init");
        Require(aclInit(nullptr), "aclInit"); runtime.initialized = true;
        Require(aclrtSetDevice(runtime.device), "aclrtSetDevice"); runtime.deviceSelected = true;
        Require(aclrtCreateContext(&runtime.context, runtime.device), "aclrtCreateContext");
        aclrtRunMode mode;
        Require(aclrtGetRunMode(&mode), "aclrtGetRunMode");
        if (mode != ACL_HOST) throw std::runtime_error("ACL_HOST required");
        Require(aclrtCreateStream(&runtime.stream), "aclrtCreateStream");
        Resources resources;
        resources.stream = runtime.stream;
        Stage("allocate_and_upload");
        void *partialMemory = resources.Allocate(inputBytes), *localMemory = resources.Allocate(inputBytes);
        void *scoreMemory = resources.Allocate(outputBytes), *idMemory = resources.Allocate(outputBytes);
        auto *partialTensor = DescribeCandidates(resources, q, shards, k, ACL_FLOAT, partialMemory);
        auto *localTensor = DescribeCandidates(resources, q, shards, k, ACL_INT32, localMemory);
        auto *scoreTensor = resources.Describe(q, k, ACL_FLOAT, scoreMemory);
        auto *idTensor = resources.Describe(q, k, ACL_INT32, idMemory);
        Require(aclrtMemcpy(partialMemory, inputBytes, partial.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload partial scores");
        Require(aclrtMemcpy(localMemory, inputBytes, local.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload local IDs");
        std::vector<float> actualScores(outputCount), baselineScores;
        std::vector<int32_t> actualIds(outputCount), baselineIds;
        std::vector<uint32_t> poison(outputCount, UINT32_MAX);
        uint64_t capacity = 0;
        void *workspace = nullptr;
        for (int iteration = 0; iteration < 3; ++iteration) {
            std::cout << "repeat=" << iteration << std::endl;
            Require(aclrtMemcpy(scoreMemory, outputBytes, poison.data(), outputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison scores");
            Require(aclrtMemcpy(idMemory, outputBytes, poison.data(), outputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison IDs");
            Stage("get_workspace_size");
            uint64_t bytes = 0;
            aclOpExecutor *executor = nullptr;
            Require(prepare(partialTensor, localTensor, k, docs, scoreTensor, idTensor, &bytes, &executor), "merge prepare");
            if (!executor || bytes > std::numeric_limits<size_t>::max()) throw std::runtime_error("Invalid executor/workspace");
            if (iteration == 0) {
                capacity = bytes; workspace = resources.Allocate(static_cast<size_t>(bytes));
                std::cout << "workspace_bytes=" << bytes << std::endl;
            } else if (bytes > capacity) throw std::runtime_error("Workspace grew during reuse");
            Stage("execute");
            Require(execute(workspace, bytes, executor, runtime.stream), "merge execute");
            executor = nullptr; // Ordinary executor is consumed by the generated execution API.
            Stage("synchronize");
            Require(aclrtSynchronizeStream(runtime.stream), "merge synchronize");
            Stage("download_and_validate");
            Require(aclrtMemcpy(actualScores.data(), outputBytes, scoreMemory, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), "download scores");
            Require(aclrtMemcpy(actualIds.data(), outputBytes, idMemory, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), "download IDs");
            ValidateSelected(partial, local, actualScores, actualIds, q, shards, docs, k);
            if (iteration == 0) { baselineScores = actualScores; baselineIds = actualIds; }
            else if (std::memcmp(actualScores.data(), baselineScores.data(), outputBytes) != 0 ||
                     std::memcmp(actualIds.data(), baselineIds.data(), outputBytes) != 0)
                throw std::runtime_error("Repeated merge output bytes changed");
        }
        Stage("release_resources");
        if (!resources.Close() || !runtime.Close()) throw std::runtime_error("Cleanup failed; no output saved");
        Stage("save_output");
        fs::create_directories(output.parent_path());
        if (!fs::create_directory(output)) throw std::runtime_error("Cannot create fresh output directory");
        Write(output / "top_scores.bin", actualScores);
        Write(output / "top_ids.bin", actualIds);
        std::cout << "device=" << device << " queries=" << q << " shards=" << shards << " docs=" << docs << " k=" << k
                  << " completed_poisoned_repeats=3\nMerge unit execution completed; independent ranking check still required. Not a BM25 performance benchmark.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
