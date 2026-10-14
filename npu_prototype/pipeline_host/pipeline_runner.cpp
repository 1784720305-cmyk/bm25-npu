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

int main(int argc, char **argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-libraries") {
            Interfaces api(fs::absolute(argv[2]));
            std::cout << "All six generated API symbols belong to the requested libraries. No device work submitted.\n";
            return 0;
        }
        if (argc != 8) {
            std::cerr << "Usage: bm25_pipeline_runner INPUT_DIR OUTPUT_DIR QUERIES DOCS K DEVICE VENDOR_DIR\n"
                      << "       bm25_pipeline_runner --check-libraries VENDOR_DIR\n";
            return 2;
        }
        static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4 && sizeof(int64_t) == 8);
        static_assert(std::numeric_limits<float>::is_iec559);
        const uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian) != 1) throw std::runtime_error("Little-endian host required");
        Stage("validate_inputs");
        const int64_t queries = ParseInteger(argv[3], "query count", 1, INT32_MAX);
        const int64_t docs = ParseInteger(argv[4], "document count", 1, INT32_MAX);
        const int64_t k = ParseInteger(argv[5], "K", 1, 32);
        const int64_t device = ParseInteger(argv[6], "device", 0, INT32_MAX);
        const uint64_t groupCount = static_cast<uint64_t>(queries) * static_cast<uint64_t>(docs);
        if (groupCount > INT32_MAX || k > docs) throw std::runtime_error("Group count or K exceeds operator limits");
        const auto inputDir = fs::absolute(argv[1]);
        const auto outputDir = fs::absolute(argv[2]);
        if (fs::exists(outputDir)) throw std::runtime_error("Choose a fresh output directory");
        const auto tf = Read<float>(inputDir / "tf.bin");
        const auto norm = Read<float>(inputDir / "doc_norm.bin");
        const auto idf = Read<float>(inputDir / "idf.bin");
        const auto offsets = Read<int64_t>(inputDir / "offsets.bin");
        if (tf.size() > INT32_MAX || norm.size() != tf.size() || idf.size() != tf.size() ||
            offsets.size() != groupCount + 1 || offsets.front() != 0 || offsets.back() != static_cast<int64_t>(tf.size()))
            throw std::runtime_error("Input lengths or endpoint offsets are invalid");
        for (size_t i = 0; i < tf.size(); ++i) {
            if (!std::isfinite(tf[i]) || !std::isfinite(norm[i]) || !std::isfinite(idf[i]) ||
                tf[i] <= 0 || tf[i] > 16777216.0F || std::floor(tf[i]) != tf[i] || norm[i] < 0 || idf[i] < 0)
                throw std::runtime_error("Invalid TF/norm/IDF record");
        }
        for (size_t i = 0; i < offsets.size(); ++i) {
            if (offsets[i] < 0 || offsets[i] > static_cast<int64_t>(tf.size()) || (i && offsets[i] < offsets[i-1]))
                throw std::runtime_error("Offsets out of bounds or decreasing");
        }
        const size_t valueBytes = CheckedBytes(tf.size(), sizeof(float));
        const size_t offsetBytes = CheckedBytes(offsets.size(), sizeof(int64_t));
        const size_t scoreBytes = CheckedBytes(groupCount, sizeof(float));
        const uint64_t topCount = static_cast<uint64_t>(queries) * static_cast<uint64_t>(k);
        const size_t topBytes = CheckedBytes(topCount, sizeof(float));
        std::cout << "value_count=" << tf.size() << " group_count=" << groupCount << " queries=" << queries
                  << " docs=" << docs << " k=" << k << std::endl;
        Stage("load_three_api_libraries");
        Interfaces api(fs::absolute(argv[7]));
        Runtime runtime;
        runtime.device = static_cast<int>(device);
        Stage("aclInit");
        Require(aclInit(nullptr), "aclInit"); runtime.initialized = true;
        Stage("aclrtSetDevice");
        Require(aclrtSetDevice(runtime.device), "aclrtSetDevice"); runtime.deviceSelected = true;
        Stage("aclrtCreateContext");
        Require(aclrtCreateContext(&runtime.context, runtime.device), "aclrtCreateContext");
        aclrtRunMode mode;
        Require(aclrtGetRunMode(&mode), "aclrtGetRunMode");
        if (mode != ACL_HOST) throw std::runtime_error("ACL_HOST mode required");
        Stage("aclrtCreateStream");
        Require(aclrtCreateStream(&runtime.stream), "aclrtCreateStream");
        Resources resources;
        resources.stream = runtime.stream;
        Stage("allocate_and_describe");
        void *inputs[3];
        aclTensor *inputTensors[3];
        for (int i = 0; i < 3; ++i) {
            inputs[i] = resources.Allocate(valueBytes);
            inputTensors[i] = resources.Describe1D(static_cast<int64_t>(tf.size()), ACL_FLOAT, inputs[i]);
        }
        void *offsetMemory = resources.Allocate(offsetBytes);
        auto *offsetTensor = resources.Describe1D(static_cast<int64_t>(offsets.size()), ACL_INT64, offsetMemory);
        void *contributionMemory = resources.Allocate(valueBytes);
        auto *contributionTensor = resources.Describe1D(static_cast<int64_t>(tf.size()), ACL_FLOAT, contributionMemory);
        void *scoreMemory = resources.Allocate(scoreBytes);
        auto *flatScoreTensor = resources.Describe1D(static_cast<int64_t>(groupCount), ACL_FLOAT, scoreMemory);
        // Both tensor descriptions share the exact same device score allocation; no data copy/reshape kernel.
        auto *matrixScoreTensor = resources.Describe(queries, docs, ACL_FLOAT, scoreMemory);
        void *topScoreMemory = resources.Allocate(topBytes);
        void *topIdMemory = resources.Allocate(topBytes);
        auto *topScoreTensor = resources.Describe(queries, k, ACL_FLOAT, topScoreMemory);
        auto *topIdTensor = resources.Describe(queries, k, ACL_INT32, topIdMemory);
        std::vector<float> actualContribution(tf.size(), std::numeric_limits<float>::quiet_NaN());
        std::vector<float> actualScores(static_cast<size_t>(groupCount), std::numeric_limits<float>::quiet_NaN());
        std::vector<float> actualTop(static_cast<size_t>(topCount), std::numeric_limits<float>::quiet_NaN());
        std::vector<int32_t> actualIds(static_cast<size_t>(topCount), -1);
        Require(aclrtMemcpy(contributionMemory, valueBytes, actualContribution.data(), valueBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison contributions");
        Require(aclrtMemcpy(scoreMemory, scoreBytes, actualScores.data(), scoreBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison full scores");
        Require(aclrtMemcpy(topScoreMemory, topBytes, actualTop.data(), topBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison top scores");
        Require(aclrtMemcpy(topIdMemory, topBytes, actualIds.data(), topBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison top IDs");
        const auto start = std::chrono::steady_clock::now();
        Stage("upload_inputs_once");
        const std::vector<float> *hostInputs[3] = {&tf, &norm, &idf};
        for (int i = 0; i < 3; ++i)
            Require(aclrtMemcpy(inputs[i], valueBytes, hostInputs[i]->data(), valueBytes, ACL_MEMCPY_HOST_TO_DEVICE), "input upload");
        Require(aclrtMemcpy(offsetMemory, offsetBytes, offsets.data(), offsetBytes, ACL_MEMCPY_HOST_TO_DEVICE), "offset upload");
        uint64_t workspaceSize[3] = {0, 0, 0};
        aclOpExecutor *executor[3] = {nullptr, nullptr, nullptr};
        // Own tiling functions depend only on shapes/attributes; all shapes are known before execution.
        Stage("prepare_contribution");
        Require(api.prepareContribution(inputTensors[0], inputTensors[1], inputTensors[2], contributionTensor,
                                        &workspaceSize[0], &executor[0]), "contribution prepare");
        Stage("prepare_accumulate");
        Require(api.prepareAccumulate(contributionTensor, offsetTensor, flatScoreTensor,
                                      &workspaceSize[1], &executor[1]), "accumulate prepare");
        Stage("prepare_topk");
        Require(api.prepareTopK(matrixScoreTensor, k, topScoreTensor, topIdTensor,
                               &workspaceSize[2], &executor[2]), "Top-K prepare");
        void *workspace[3];
        for (int i = 0; i < 3; ++i) {
            if (!executor[i] || workspaceSize[i] > std::numeric_limits<size_t>::max()) throw std::runtime_error("Invalid executor/workspace");
            workspace[i] = resources.Allocate(static_cast<size_t>(workspaceSize[i]));
            std::cout << "workspace_stage=" << i << " bytes=" << workspaceSize[i] << std::endl;
        }
        // Three distinct workspaces retained until completion; no assumed cross-operator scratch reuse.
        const auto launchStart = std::chrono::steady_clock::now();
        Stage("execute_contribution");
        Require(api.executeContribution(workspace[0], workspaceSize[0], executor[0], runtime.stream), "contribution execute"); executor[0] = nullptr;
        Stage("execute_accumulate");
        Require(api.executeAccumulate(workspace[1], workspaceSize[1], executor[1], runtime.stream), "accumulate execute"); executor[1] = nullptr;
        Stage("execute_topk");
        Require(api.executeTopK(workspace[2], workspaceSize[2], executor[2], runtime.stream), "Top-K execute"); executor[2] = nullptr;
        Stage("synchronize_pipeline");
        Require(aclrtSynchronizeStream(runtime.stream), "pipeline synchronize");
        const auto launchEnd = std::chrono::steady_clock::now();
        Stage("download_after_all_three");
        Require(aclrtMemcpy(actualTop.data(), topBytes, topScoreMemory, topBytes, ACL_MEMCPY_DEVICE_TO_HOST), "top score download");
        Require(aclrtMemcpy(actualIds.data(), topBytes, topIdMemory, topBytes, ACL_MEMCPY_DEVICE_TO_HOST), "top ID download");
        // Additional downloads only after the entire device chain, for first-run correctness diagnostics.
        Require(aclrtMemcpy(actualContribution.data(), valueBytes, contributionMemory, valueBytes, ACL_MEMCPY_DEVICE_TO_HOST), "debug contribution download");
        Require(aclrtMemcpy(actualScores.data(), scoreBytes, scoreMemory, scoreBytes, ACL_MEMCPY_DEVICE_TO_HOST), "debug full score download");
        const auto end = std::chrono::steady_clock::now();
        for (float value : actualContribution) if (!std::isfinite(value) || value < 0) throw std::runtime_error("Invalid device contribution");
        for (float value : actualScores) if (!std::isfinite(value) || value < 0) throw std::runtime_error("Invalid device full score");
        ValidateOutput(actualScores, actualTop, actualIds, queries, docs, k);
        Stage("release_resources");
        if (!resources.Close() || !runtime.Close()) throw std::runtime_error("Cleanup failed; results not saved");
        Stage("save_output");
        fs::create_directories(outputDir.parent_path());
        if (!fs::create_directory(outputDir)) throw std::runtime_error("Cannot create fresh output directory");
        try {
            Write(outputDir / "contributions.bin", actualContribution);
            Write(outputDir / "scores.bin", actualScores);
            Write(outputDir / "top_scores.bin", actualTop);
            Write(outputDir / "top_ids.bin", actualIds);
        } catch (...) {
            std::error_code ignored;
            for (const char *name : {"contributions.bin", "scores.bin", "top_scores.bin", "top_ids.bin"}) fs::remove(outputDir / name, ignored);
            throw;
        }
        std::cout << "device=" << device << " pipeline_stages=3 intermediate_host_roundtrips=0 output=" << outputDir << '\n'
                  << "single_three_launches_and_sync_ms=" << std::chrono::duration<double, std::milli>(launchEnd-launchStart).count() << '\n'
                  << "single_upload_prepare_execute_debug_download_ms=" << std::chrono::duration<double, std::milli>(end-start).count() << '\n'
                  << "First-run device-pipeline diagnostic; includes final debug downloads. Not steady-state or complete retrieval benchmark.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
