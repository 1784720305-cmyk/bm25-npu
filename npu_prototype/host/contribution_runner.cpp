// Direct generated ACLNN interface; signatures verified from the A5 installed header.
// Not a CPU fallback: execution failures abort without writing computed results.
#include "acl/acl.h"
#include "aclnn_bm25_contribution.h"
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
                                        const aclTensor *, uint64_t *, aclOpExecutor **);
using ExecuteSignature = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
static_assert(std::is_same<decltype(&aclnnBm25ContributionGetWorkspaceSize), PrepareSignature>::value,
              "Generated prepare signature differs from the A5 header provided");
static_assert(std::is_same<decltype(&aclnnBm25Contribution), ExecuteSignature>::value,
              "Generated execute signature differs from the A5 header provided");

void Stage(const char *name) { std::cout << "stage=" << name << std::endl; }

void Require(aclError error, const char *operation)
{
    if (error == ACL_ERROR_NONE) return;
    const char *detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(operation) + " failed, code=" + std::to_string(error) +
                             ", detail=" + (detail == nullptr ? "<none>" : detail));
}

std::vector<float> Read(const fs::path &path)
{
    const auto bytes = fs::file_size(path);
    if (bytes == 0 || bytes % sizeof(float) != 0 || bytes / sizeof(float) > INT32_MAX)
        throw std::runtime_error("Invalid FP32 input size: " + path.string());
    std::vector<float> result(bytes / sizeof(float));
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("Cannot read: " + path.string());
    return result;
}

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
                return false; // Do not tear down objects while device work may still be pending.
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
        // Wait before freeing tensors/workspace referenced by submitted device work.
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
    aclTensor *Describe(int64_t count, void *deviceData)
    {
        const int64_t stride = 1;
        auto *tensor = aclCreateTensor(&count, 1, ACL_FLOAT, &stride, 0, ACL_FORMAT_ND,
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
            std::cerr << "Usage: runner INPUT_DIR ACTUAL_BIN DEVICE_ID\n";
            return 2;
        }
        static_assert(sizeof(float) == 4);
        const uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian) != 1)
            throw std::runtime_error("This fixture requires little-endian host");
        const auto inputDir = fs::absolute(argv[1]);
        const auto outputPath = fs::absolute(argv[2]);
        if (fs::exists(outputPath)) throw std::runtime_error("Output already exists; choose a fresh output name");
        auto tf = Read(inputDir / "tf.bin");
        auto norm = Read(inputDir / "doc_norm.bin");
        auto idf = Read(inputDir / "idf.bin");
        if (tf.size() != norm.size() || tf.size() != idf.size())
            throw std::runtime_error("Input lengths differ");
        for (size_t i = 0; i < tf.size(); ++i) {
            if (!std::isfinite(tf[i]) || !std::isfinite(norm[i]) || !std::isfinite(idf[i]) ||
                tf[i] <= 0 || tf[i] > 16777216.0F || std::floor(tf[i]) != tf[i] || norm[i] < 0 || idf[i] < 0)
                throw std::runtime_error("Invalid TF/norm/IDF at record " + std::to_string(i));
        }
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
        const size_t bytes = tf.size() * sizeof(float);
        const int64_t count = static_cast<int64_t>(tf.size());
        Stage("allocate_and_create_tensors");
        void *inputMemory[3];
        aclTensor *inputs[3];
        const std::vector<float> *hostInputs[3] = {&tf, &norm, &idf};
        for (int i = 0; i < 3; ++i) {
            inputMemory[i] = resources.Allocate(bytes);
            inputs[i] = resources.Describe(count, inputMemory[i]);
        }
        void *outputMemory = resources.Allocate(bytes);
        aclTensor *outputTensor = resources.Describe(count, outputMemory);
        // Poison output to catch omitted writes, rather than accepting old/zero memory.
        std::vector<float> poison(tf.size(), std::numeric_limits<float>::quiet_NaN());
        Require(aclrtMemcpy(outputMemory, bytes, poison.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "output initialization");
        // Includes input H2D, prepare/workspace allocation, execution/sync, D2H. Diagnostic only.
        auto start = std::chrono::steady_clock::now();
        Stage("upload_inputs");
        for (int i = 0; i < 3; ++i)
            Require(aclrtMemcpy(inputMemory[i], bytes, hostInputs[i]->data(), bytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE), "input upload");
        Stage("get_workspace_size");
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        Require(aclnnBm25ContributionGetWorkspaceSize(inputs[0], inputs[1], inputs[2], outputTensor,
                                                      &workspaceSize, &executor), "aclnn prepare");
        if (executor == nullptr) throw std::runtime_error("Prepare returned null executor");
        if (workspaceSize > std::numeric_limits<size_t>::max())
            throw std::runtime_error("Workspace size exceeds host size_t");
        std::cout << "workspace_bytes=" << workspaceSize << std::endl;
        void *workspace = resources.Allocate(static_cast<size_t>(workspaceSize));
        Stage("execute");
        auto launchStart = std::chrono::steady_clock::now();
        // Non-repeatable executor is framework-owned and consumed by the second-stage API.
        // Do not manually delete it or call a repeatable-executor destroy API.
        Require(aclnnBm25Contribution(workspace, workspaceSize, executor, runtime.stream), "aclnn execute");
        executor = nullptr;
        Stage("synchronize");
        Require(aclrtSynchronizeStream(runtime.stream), "aclrtSynchronizeStream");
        auto launchEnd = std::chrono::steady_clock::now();
        Stage("download_output");
        std::vector<float> actual(tf.size());
        Require(aclrtMemcpy(actual.data(), bytes, outputMemory, bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST), "output download");
        auto end = std::chrono::steady_clock::now();
        for (float value : actual)
            if (!std::isfinite(value)) throw std::runtime_error("Device output contains non-finite values");
        Stage("release_resources");
        if (!resources.Close()) throw std::runtime_error("Resource cleanup failed; output not saved");
        if (!runtime.Close()) throw std::runtime_error("Runtime cleanup failed; output not saved");
        Stage("save_output");
        fs::create_directories(outputPath.parent_path());
        std::ofstream output(outputPath, std::ios::binary);
        output.write(reinterpret_cast<const char *>(actual.data()), static_cast<std::streamsize>(bytes));
        output.close();
        if (!output) throw std::runtime_error("Cannot save output");
        std::cout << "device=" << runtime.device << " records=" << count << " output=" << outputPath << '\n'
                  << "single_launch_and_sync_ms=" << std::chrono::duration<double, std::milli>(launchEnd-launchStart).count() << '\n'
                  << "single_upload_prepare_execute_download_ms=" << std::chrono::duration<double, std::milli>(end-start).count() << '\n'
                  << "ACLNN diagnostic only; validate output separately. Not steady-state BM25 benchmark.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
