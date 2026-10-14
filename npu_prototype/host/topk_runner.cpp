// Direct signatures confirmed from the installed A5 Top-K header.
// Reads only scores.bin; failures stop without CPU ranking/fallback.
#include "acl/acl.h"
#include "aclnn_bm25_top_k.h"
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

int main(int argc, char **argv)
{
    try {
        if (argc != 7) {
            std::cerr << "Usage: bm25_topk_runner INPUT_DIR OUTPUT_DIR QUERY_COUNT DOC_COUNT K DEVICE_ID\n";
            return 2;
        }
        static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4);
        static_assert(std::numeric_limits<float>::is_iec559);
        const uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian) != 1)
            throw std::runtime_error("This fixture requires little-endian host");
        Stage("validate_inputs");
        const int64_t queries = ParseInteger(argv[3], "query count", 1, INT32_MAX);
        const int64_t docs = ParseInteger(argv[4], "document count", 1, INT32_MAX);
        const int64_t k = ParseInteger(argv[5], "K", 1, 32);
        if (k > docs) throw std::runtime_error("K exceeds document count; no automatic clamping");
        const uint64_t inputCount = static_cast<uint64_t>(queries) * static_cast<uint64_t>(docs);
        const uint64_t outputCount = static_cast<uint64_t>(queries) * static_cast<uint64_t>(k);
        const size_t inputBytes = CheckedBytes(inputCount, sizeof(float));
        const size_t scoreBytes = CheckedBytes(outputCount, sizeof(float));
        const size_t idBytes = CheckedBytes(outputCount, sizeof(int32_t));
        const auto inputDir = fs::absolute(argv[1]);
        const auto outputDir = fs::absolute(argv[2]);
        if (fs::exists(outputDir)) throw std::runtime_error("Output directory already exists; choose a fresh name");
        const auto inputPath = inputDir / "scores.bin";
        if (fs::file_size(inputPath) != inputBytes) throw std::runtime_error("Input bytes do not match query/document counts");
        const auto input = Read<float>(inputPath);
        for (size_t i = 0; i < input.size(); ++i) {
            if (!std::isfinite(input[i]) || input[i] < 0)
                throw std::runtime_error("Input score must be finite and nonnegative at record " + std::to_string(i));
        }
        std::cout << "query_count=" << queries << " doc_count=" << docs << " k=" << k
                  << " output_count=" << outputCount << std::endl;
        Runtime runtime;
        runtime.device = static_cast<int>(ParseInteger(argv[6], "device ID", 0, INT32_MAX));
        Stage("runtime_init");
        Require(aclInit(nullptr), "aclInit");
        runtime.initialized = true;
        Require(aclrtSetDevice(runtime.device), "aclrtSetDevice");
        runtime.deviceSelected = true;
        Require(aclrtCreateContext(&runtime.context, runtime.device), "aclrtCreateContext");
        aclrtRunMode mode;
        Require(aclrtGetRunMode(&mode), "aclrtGetRunMode");
        if (mode != ACL_HOST) throw std::runtime_error("This diagnostic requires ACL_HOST mode");
        Require(aclrtCreateStream(&runtime.stream), "aclrtCreateStream");
        Resources resources;
        resources.stream = runtime.stream;
        Stage("allocate_and_create_tensors");
        void *inputMemory = resources.Allocate(inputBytes);
        void *scoreMemory = resources.Allocate(scoreBytes);
        void *idMemory = resources.Allocate(idBytes);
        aclTensor *inputTensor = resources.Describe(queries, docs, ACL_FLOAT, inputMemory);
        aclTensor *scoreTensor = resources.Describe(queries, k, ACL_FLOAT, scoreMemory);
        aclTensor *idTensor = resources.Describe(queries, k, ACL_INT32, idMemory);
        std::vector<float> actualScores(static_cast<size_t>(outputCount), std::numeric_limits<float>::quiet_NaN());
        std::vector<int32_t> actualIds(static_cast<size_t>(outputCount), -1);
        // Detect omitted writes in both outputs, including otherwise-valid zero scores.
        Require(aclrtMemcpy(scoreMemory, scoreBytes, actualScores.data(), scoreBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "score output initialization");
        Require(aclrtMemcpy(idMemory, idBytes, actualIds.data(), idBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "ID output initialization");
        const auto start = std::chrono::steady_clock::now();
        Stage("upload_inputs");
        Require(aclrtMemcpy(inputMemory, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "score input upload");
        Stage("get_workspace_size");
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        Require(aclnnBm25TopKGetWorkspaceSize(inputTensor, k, scoreTensor, idTensor,
                                            &workspaceSize, &executor), "aclnn prepare");
        if (executor == nullptr) throw std::runtime_error("Prepare returned null executor");
        if (workspaceSize > std::numeric_limits<size_t>::max()) throw std::runtime_error("Workspace exceeds host limits");
        std::cout << "workspace_bytes=" << workspaceSize << std::endl;
        void *workspace = resources.Allocate(static_cast<size_t>(workspaceSize));
        Stage("execute");
        const auto launchStart = std::chrono::steady_clock::now();
        // Framework-owned, non-repeatable executor is consumed by the second-stage API.
        // Failures terminate this diagnostic process; no manual repeatable-executor destruction.
        Require(aclnnBm25TopK(workspace, workspaceSize, executor, runtime.stream), "aclnn execute");
        executor = nullptr;
        Stage("synchronize");
        Require(aclrtSynchronizeStream(runtime.stream), "aclrtSynchronizeStream");
        const auto launchEnd = std::chrono::steady_clock::now();
        Stage("download_output");
        Require(aclrtMemcpy(actualScores.data(), scoreBytes, scoreMemory, scoreBytes, ACL_MEMCPY_DEVICE_TO_HOST), "score download");
        Require(aclrtMemcpy(actualIds.data(), idBytes, idMemory, idBytes, ACL_MEMCPY_DEVICE_TO_HOST), "ID download");
        const auto end = std::chrono::steady_clock::now();
        ValidateOutput(input, actualScores, actualIds, queries, docs, k);
        Stage("release_resources");
        if (!resources.Close()) throw std::runtime_error("Resource cleanup failed; output not saved");
        if (!runtime.Close()) throw std::runtime_error("Runtime cleanup failed; output not saved");
        Stage("save_output");
        fs::create_directories(outputDir.parent_path());
        if (!fs::create_directory(outputDir)) throw std::runtime_error("Cannot create fresh output directory");
        try {
            Write(outputDir / "top_scores.bin", actualScores);
            Write(outputDir / "top_ids.bin", actualIds);
        } catch (...) {
            // Remove only newly-written, fixed-name files owned by this invocation, never recursively.
            std::error_code ignored;
            fs::remove(outputDir / "top_scores.bin", ignored);
            fs::remove(outputDir / "top_ids.bin", ignored);
            throw;
        }
        std::cout << "device=" << runtime.device << " queries=" << queries << " k=" << k << " output=" << outputDir << '\n'
                  << "single_launch_and_sync_ms=" << std::chrono::duration<double, std::milli>(launchEnd-launchStart).count() << '\n'
                  << "single_upload_prepare_execute_download_ms=" << std::chrono::duration<double, std::milli>(end-start).count() << '\n'
                  << "ACLNN Top-K diagnostic only; validate full ranking separately. Not a BM25 benchmark.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
