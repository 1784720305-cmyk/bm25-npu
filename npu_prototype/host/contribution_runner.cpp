// Minimal standalone host diagnostic using IndexSDK's ACL handle execution path.
// Not a CPU fallback: execution failures abort without writing computed results.
#include "acl/acl.h"
#include "acl/acl_op.h"
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
    ~Runtime()
    {
        if (stream != nullptr) { aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); }
        if (context != nullptr) aclrtDestroyContext(context);
        if (deviceSelected) aclrtResetDevice(device);
        if (initialized) aclFinalize();
    }
};

struct Resources {
    aclrtStream stream = nullptr;
    std::vector<void *> memory;
    std::vector<aclDataBuffer *> buffers;
    std::vector<aclTensorDesc *> descriptors;
    aclopAttr *attributes = nullptr;
    aclopHandle *handle = nullptr;
    ~Resources()
    {
        // Wait even on an exception before releasing memory referenced by an outstanding op.
        if (stream != nullptr) aclrtSynchronizeStream(stream);
        if (handle != nullptr) aclopDestroyHandle(handle);
        if (attributes != nullptr) aclopDestroyAttr(attributes);
        for (auto *p : buffers) aclDestroyDataBuffer(p);
        for (auto *p : descriptors) aclDestroyTensorDesc(p);
        for (auto *p : memory) aclrtFree(p);
    }
    aclDataBuffer *Allocate(size_t bytes)
    {
        void *p = nullptr;
        Require(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
        memory.push_back(p);
        auto *buffer = aclCreateDataBuffer(p, bytes);
        if (buffer == nullptr) throw std::runtime_error("aclCreateDataBuffer returned null");
        buffers.push_back(buffer);
        return buffer;
    }
    const aclTensorDesc *Describe(int64_t count)
    {
        auto *desc = aclCreateTensorDesc(ACL_FLOAT, 1, &count, ACL_FORMAT_ND);
        if (desc == nullptr) throw std::runtime_error("aclCreateTensorDesc returned null");
        descriptors.push_back(desc);
        return desc;
    }
};

int main(int argc, char **argv)
{
    try {
        if (argc != 5) {
            std::cerr << "Usage: runner MODEL_DIR INPUT_DIR ACTUAL_BIN DEVICE_ID\n";
            return 2;
        }
        static_assert(sizeof(float) == 4);
        const uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian) != 1)
            throw std::runtime_error("This fixture requires little-endian host");
        const auto modelDir = fs::absolute(argv[1]);
        const auto inputDir = fs::absolute(argv[2]);
        const auto outputPath = fs::absolute(argv[3]);
        if (!fs::is_directory(modelDir)) throw std::runtime_error("Model directory missing");
        bool hasModel = false;
        for (const auto &entry : fs::directory_iterator(modelDir))
            hasModel |= entry.is_regular_file() && entry.path().extension() == ".om";
        if (!hasModel) throw std::runtime_error("No OM model in model directory");
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
        runtime.device = std::stoi(argv[4], &used);
        if (used != std::string(argv[4]).size() || runtime.device < 0)
            throw std::runtime_error("Invalid device ID");
        Require(aclInit(nullptr), "aclInit");
        runtime.initialized = true;
        Require(aclrtSetDevice(runtime.device), "aclrtSetDevice");
        runtime.deviceSelected = true;
        Require(aclrtCreateContext(&runtime.context, runtime.device), "aclrtCreateContext");
        aclrtRunMode mode;
        Require(aclrtGetRunMode(&mode), "aclrtGetRunMode");
        if (mode != ACL_HOST) throw std::runtime_error("This host diagnostic requires ACL_HOST mode");
        Require(aclrtCreateStream(&runtime.stream), "aclrtCreateStream");
        Require(aclopSetModelDir(modelDir.string().c_str()), "aclopSetModelDir");
        Resources resources;
        resources.stream = runtime.stream;
        const size_t bytes = tf.size() * sizeof(float);
        const int64_t count = static_cast<int64_t>(tf.size());
        const aclTensorDesc *inputDesc[3];
        const aclDataBuffer *inputs[3];
        const std::vector<float> *hostInputs[3] = {&tf, &norm, &idf};
        for (int i = 0; i < 3; ++i) {
            inputDesc[i] = resources.Describe(count);
            inputs[i] = resources.Allocate(bytes);
        }
        const aclTensorDesc *outputDesc[1] = {resources.Describe(count)};
        aclDataBuffer *outputs[1] = {resources.Allocate(bytes)};
        resources.attributes = aclopCreateAttr();
        if (resources.attributes == nullptr) throw std::runtime_error("aclopCreateAttr returned null");
        Require(aclopCreateHandle("Bm25Contribution", 3, inputDesc, 1, outputDesc,
                                  resources.attributes, &resources.handle), "aclopCreateHandle");
        // Includes H2D, one execution, synchronization and D2H. Not a warmed benchmark.
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 3; ++i)
            Require(aclrtMemcpy(resources.memory[i], bytes, hostInputs[i]->data(), bytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE), "input upload");
        auto launchStart = std::chrono::steady_clock::now();
        Require(aclopExecWithHandle(resources.handle, 3, inputs, 1, outputs, runtime.stream), "aclopExecWithHandle");
        Require(aclrtSynchronizeStream(runtime.stream), "aclrtSynchronizeStream");
        auto launchEnd = std::chrono::steady_clock::now();
        std::vector<float> actual(tf.size());
        Require(aclrtMemcpy(actual.data(), bytes, resources.memory[3], bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST), "output download");
        auto end = std::chrono::steady_clock::now();
        for (float value : actual)
            if (!std::isfinite(value)) throw std::runtime_error("Device output contains non-finite values");
        fs::create_directories(outputPath.parent_path());
        std::ofstream output(outputPath, std::ios::binary);
        output.write(reinterpret_cast<const char *>(actual.data()), static_cast<std::streamsize>(bytes));
        output.close();
        if (!output) throw std::runtime_error("Cannot save output");
        std::cout << "device=" << runtime.device << " records=" << count << " output=" << outputPath << '\n'
                  << "single_launch_and_sync_ms=" << std::chrono::duration<double, std::milli>(launchEnd-launchStart).count() << '\n'
                  << "single_upload_execute_download_ms=" << std::chrono::duration<double, std::milli>(end-start).count() << '\n'
                  << "Diagnostic run only; validate output separately. Not steady-state BM25 benchmark.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
