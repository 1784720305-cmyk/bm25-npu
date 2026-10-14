// Direct ACLNN signatures confirmed from the installed A5 accumulation header.
// Reads only values/offsets, never expected.bin; failures stop without CPU fallback.
#include "acl/acl.h"
#include "aclnn_bm25_accumulate.h"
#include <type_traits>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using PrepareSignature = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *,
                                        uint64_t *, aclOpExecutor **);
using ExecuteSignature = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
static_assert(std::is_same<decltype(&aclnnBm25AccumulateGetWorkspaceSize), PrepareSignature>::value,
              "Generated prepare signature differs from the A5 header provided");
static_assert(std::is_same<decltype(&aclnnBm25Accumulate), ExecuteSignature>::value,
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

void ValidateInputs(const std::vector<float> &values, const std::vector<int64_t> &offsets)
{
    // Current operator needs a nonempty physical values tensor. No CPU all-empty shortcut.
    if (values.empty() || offsets.size() < 2 ||
        offsets.size() - 1 > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("Invalid value/group count for this operator");
    const auto valueCount = static_cast<int64_t>(values.size());
    if (offsets.front() != 0 || offsets.back() != valueCount)
        throw std::runtime_error("Offsets must start at 0 and end at value_count");
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (offsets[i] < 0 || offsets[i] > valueCount || (i != 0 && offsets[i] < offsets[i - 1]))
            throw std::runtime_error("Invalid group offset at index " + std::to_string(i));
    }
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i]) || values[i] < 0)
            throw std::runtime_error("Contribution must be finite and nonnegative at index " + std::to_string(i));
    }
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
    aclTensor *Describe(int64_t count, aclDataType type, void *deviceData)
    {
        const int64_t stride = 1;
        auto *tensor = aclCreateTensor(&count, 1, type, &stride, 0, ACL_FORMAT_ND,
                                      &count, 1, deviceData);
        if (tensor == nullptr) throw std::runtime_error("aclCreateTensor returned null");
        tensors.push_back(tensor);
        return tensor;
    }
};

int main(int argc, char **argv)
{
    try {
        if (argc != 4) {
            std::cerr << "Usage: bm25_accumulate_runner INPUT_DIR ACTUAL_BIN DEVICE_ID\n";
            return 2;
        }
        static_assert(sizeof(float) == 4 && sizeof(int64_t) == 8);
        const uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian) != 1)
            throw std::runtime_error("This fixture requires little-endian host");
        const auto inputDir = fs::absolute(argv[1]);
        const auto outputPath = fs::absolute(argv[2]);
        if (fs::exists(outputPath)) throw std::runtime_error("Output already exists; choose a fresh output name");
        Stage("validate_inputs");
        const auto values = Read<float>(inputDir / "values.bin");
        const auto offsets = Read<int64_t>(inputDir / "offsets.bin");
        ValidateInputs(values, offsets);
        const size_t groupCount = offsets.size() - 1;
        const size_t valueBytes = values.size() * sizeof(float);
        const size_t offsetBytes = offsets.size() * sizeof(int64_t);
        const size_t scoreBytes = groupCount * sizeof(float);
        std::cout << "value_count=" << values.size() << " offset_count=" << offsets.size()
                  << " group_count=" << groupCount << std::endl;
        Runtime runtime;
        size_t used = 0;
        runtime.device = std::stoi(argv[3], &used);
        if (used != std::string(argv[3]).size() || runtime.device < 0)
            throw std::runtime_error("Invalid device ID");
        Stage("runtime_init");
        Require(aclInit(nullptr), "aclInit");
        runtime.initialized = true;
        Require(aclrtSetDevice(runtime.device), "aclrtSetDevice");
        runtime.deviceSelected = true;
        Require(aclrtCreateContext(&runtime.context, runtime.device), "aclrtCreateContext");
        aclrtRunMode mode;
        Require(aclrtGetRunMode(&mode), "aclrtGetRunMode");
        if (mode != ACL_HOST) throw std::runtime_error("This host diagnostic requires ACL_HOST mode");
        Require(aclrtCreateStream(&runtime.stream), "aclrtCreateStream");
        Resources resources;
        resources.stream = runtime.stream;
        Stage("allocate_and_create_tensors");
        void *valueMemory = resources.Allocate(valueBytes);
        void *offsetMemory = resources.Allocate(offsetBytes);
        void *scoreMemory = resources.Allocate(scoreBytes);
        aclTensor *valueTensor = resources.Describe(static_cast<int64_t>(values.size()), ACL_FLOAT, valueMemory);
        aclTensor *offsetTensor = resources.Describe(static_cast<int64_t>(offsets.size()), ACL_INT64, offsetMemory);
        aclTensor *scoreTensor = resources.Describe(static_cast<int64_t>(groupCount), ACL_FLOAT, scoreMemory);
        std::vector<float> actual(groupCount, std::numeric_limits<float>::quiet_NaN());
        Require(aclrtMemcpy(scoreMemory, scoreBytes, actual.data(), scoreBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "output initialization");
        // Diagnostic only: input transfer + preparation/allocation + launch/sync + result transfer.
        const auto start = std::chrono::steady_clock::now();
        Stage("upload_inputs");
        Require(aclrtMemcpy(valueMemory, valueBytes, values.data(), valueBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "values upload");
        Require(aclrtMemcpy(offsetMemory, offsetBytes, offsets.data(), offsetBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "offsets upload");
        Stage("get_workspace_size");
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        Require(aclnnBm25AccumulateGetWorkspaceSize(valueTensor, offsetTensor, scoreTensor,
                                                   &workspaceSize, &executor), "aclnn prepare");
        if (executor == nullptr) throw std::runtime_error("Prepare returned null executor");
        if (workspaceSize > std::numeric_limits<size_t>::max())
            throw std::runtime_error("Workspace size exceeds host size_t");
        std::cout << "workspace_bytes=" << workspaceSize << std::endl;
        void *workspace = resources.Allocate(static_cast<size_t>(workspaceSize));
        Stage("execute");
        const auto launchStart = std::chrono::steady_clock::now();
        // Framework-owned non-repeatable executor is consumed by the second-stage API.
        // A prepare/allocation/execute failure ends this diagnostic process; no manual executor deletion.
        Require(aclnnBm25Accumulate(workspace, workspaceSize, executor, runtime.stream), "aclnn execute");
        executor = nullptr;
        Stage("synchronize");
        Require(aclrtSynchronizeStream(runtime.stream), "aclrtSynchronizeStream");
        const auto launchEnd = std::chrono::steady_clock::now();
        Stage("download_output");
        Require(aclrtMemcpy(actual.data(), scoreBytes, scoreMemory, scoreBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                "output download");
        const auto end = std::chrono::steady_clock::now();
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || actual[i] < 0)
                throw std::runtime_error("Invalid device score at group " + std::to_string(i));
        }
        Stage("release_resources");
        if (!resources.Close()) throw std::runtime_error("Resource cleanup failed; output not saved");
        if (!runtime.Close()) throw std::runtime_error("Runtime cleanup failed; output not saved");
        Stage("save_output");
        fs::create_directories(outputPath.parent_path());
        std::ofstream output(outputPath, std::ios::binary);
        output.write(reinterpret_cast<const char *>(actual.data()), static_cast<std::streamsize>(scoreBytes));
        output.close();
        if (!output) {
            std::error_code ignored;
            fs::remove(outputPath, ignored); // Remove only the new incomplete output from this invocation.
            throw std::runtime_error("Cannot save output");
        }
        std::cout << "device=" << runtime.device << " groups=" << groupCount << " output=" << outputPath << '\n'
                  << "single_launch_and_sync_ms=" << std::chrono::duration<double, std::milli>(launchEnd-launchStart).count() << '\n'
                  << "single_upload_prepare_execute_download_ms=" << std::chrono::duration<double, std::milli>(end-start).count() << '\n'
                  << "ACLNN accumulation diagnostic only; validate output separately. Not a BM25 benchmark.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
