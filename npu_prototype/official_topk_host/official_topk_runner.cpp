// Official reference only. Uses the actual A5 header supplied by the user.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "acl/acl.h"
#include "aclnn_topk.h"
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Prepare = aclnnStatus (*)(const aclTensor *, int64_t, int64_t, bool, bool,
                               aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using Execute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
static_assert(std::is_same<decltype(&aclnnTopkGetWorkspaceSize), Prepare>::value,
              "Actual aclnnTopk preparation signature differs from the supplied A5 header");
static_assert(std::is_same<decltype(&aclnnTopk), Execute>::value,
              "Actual aclnnTopk execution signature differs from the supplied A5 header");

void Require(int64_t code, const char *what)
{
    if (code != 0) throw std::runtime_error(std::string(what) + " failed: " + std::to_string(code));
}
void Stage(const char *name) { std::cout << "stage=" << name << std::endl; }
int64_t Integer(const char *s, int64_t lo, int64_t hi)
{
    size_t used = 0;
    auto n = std::stoll(s, &used);
    if (used != std::string(s).size() || n < lo || n > hi) throw std::runtime_error("Invalid integer argument");
    return n;
}
size_t Bytes(uint64_t count, size_t width)
{
    if (count > std::numeric_limits<size_t>::max() / width ||
        count > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / width)
        throw std::runtime_error("Byte count overflow");
    return static_cast<size_t>(count) * width;
}
template<class T> void Write(const fs::path &path, const std::vector<T> &data)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(Bytes(data.size(), sizeof(T))));
    f.close();
    if (!f) throw std::runtime_error("Cannot save " + path.string());
}

// Kept separate from the device-verified custom hosts: no generated BM25 header dependency.
struct Runtime {
    bool initialized = false, selected = false;
    int device = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    bool Close()
    {
        bool ok = true;
        auto check = [&](int64_t c, const char *name) {
            if (c != 0) { std::cerr << "cleanup_failed=" << name << " code=" << c << '\n'; ok = false; }
        };
        if (stream) {
            const auto c = aclrtSynchronizeStream(stream);
            if (c != 0) { check(c, "synchronize"); return false; }
            check(aclrtDestroyStream(stream), "destroy_stream"); stream = nullptr;
        }
        if (context) { check(aclrtDestroyContext(context), "destroy_context"); context = nullptr; }
        if (selected) { check(aclrtResetDevice(device), "reset_device"); selected = false; }
        if (initialized) { check(aclFinalize(), "finalize"); initialized = false; }
        return ok;
    }
    ~Runtime() { Close(); }
};
struct Resources {
    aclrtStream stream = nullptr;
    std::vector<void *> memory;
    std::vector<aclTensor *> tensors;
    void *Allocate(size_t bytes)
    {
        if (!bytes) return nullptr;
        void *p = nullptr;
        Require(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
        memory.push_back(p); return p;
    }
    aclTensor *Tensor(int64_t rows, int64_t cols, aclDataType type, void *data)
    {
        const int64_t shape[] = {rows, cols}, strides[] = {cols, 1};
        auto *t = aclCreateTensor(shape, 2, type, strides, 0, ACL_FORMAT_ND, shape, 2, data);
        if (!t) throw std::runtime_error("aclCreateTensor returned null");
        tensors.push_back(t); return t;
    }
    bool Close()
    {
        if (stream && aclrtSynchronizeStream(stream) != 0) {
            std::cerr << "cleanup_failed=resources synchronization; buffers retained while work may be pending\n";
            return false;
        }
        bool ok = true;
        for (auto *t : tensors) if (aclDestroyTensor(t) != 0) { std::cerr << "cleanup_failed=tensor\n"; ok = false; }
        tensors.clear();
        for (auto *p : memory) if (aclrtFree(p) != 0) { std::cerr << "cleanup_failed=memory\n"; ok = false; }
        memory.clear(); stream = nullptr; return ok;
    }
    ~Resources() { Close(); }
};
struct Library {
    fs::path path;
    void *handle = nullptr;
    explicit Library(const fs::path &p) : path(fs::canonical(p))
    {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) { const char *e = dlerror(); throw std::runtime_error(std::string("dlopen libopapi: ") + (e ? e : "null")); }
        std::cout << "loaded_official_api_library=" << path << std::endl;
    }
    template<class F> F Symbol(const char *name)
    {
        dlerror(); void *p = dlsym(handle, name); const char *e = dlerror();
        if (e || !p) throw std::runtime_error(std::string("Missing official symbol: ") + name);
        Dl_info owner{};
        if (!dladdr(p, &owner) || !owner.dli_fname || !fs::equivalent(path, owner.dli_fname))
            throw std::runtime_error("Official API symbol owner differs from requested libopapi");
        std::cout << "verified_api_symbol=" << name << " owner=" << owner.dli_fname << std::endl;
        F f; static_assert(sizeof(f) == sizeof(p)); std::memcpy(&f, &p, sizeof(f)); return f;
    }
    // CANN may retain callbacks; the API library remains loaded until process exit.
};
struct Output { std::vector<float> values; std::vector<int64_t> ids; };
void Validate(const std::vector<float> &input, const Output &out, int64_t rows, int64_t cols, int64_t k)
{
    for (int64_t r = 0; r < rows; ++r) for (int64_t j = 0; j < k; ++j) {
        const size_t o = static_cast<size_t>(r*k+j);
        const auto id = out.ids[o];
        if (id < 0 || id >= cols || !std::isfinite(out.values[o]) || out.values[o] < 0)
            throw std::runtime_error("Invalid official output value/index");
        if (std::memcmp(&out.values[o], &input[static_cast<size_t>(r*cols+id)], sizeof(float)) != 0)
            throw std::runtime_error("Official selected score does not preserve original FP32 bits");
        for (int64_t t = 0; t < j; ++t) if (out.ids[static_cast<size_t>(r*k+t)] == id)
            throw std::runtime_error("Duplicate official index");
        if (j && (out.values[o] > out.values[o-1] ||
                  (out.values[o] == out.values[o-1] && id < out.ids[o-1])))
            throw std::runtime_error("Official score/ID order violates strict BM25 rule");
    }
}
bool Equal(const Output &a, const Output &b)
{
    return a.ids == b.ids && a.values.size() == b.values.size() &&
           std::memcmp(a.values.data(), b.values.data(), Bytes(a.values.size(), sizeof(float))) == 0;
}
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b-a).count(); }
double Median(std::vector<double> a)
{
    std::sort(a.begin(), a.end());
    return a[(a.size()-1)/2]; // Nearest-rank 50th percentile, matching the existing CPU/NPU reports.
}

int main(int argc, char **argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-library") {
            Library library(fs::path(argv[2])/"lib64/libopapi.so");
            library.Symbol<Prepare>("aclnnTopkGetWorkspaceSize");
            library.Symbol<Execute>("aclnnTopk");
            std::cout << "Official API library and both symbol owners verified. No device work submitted.\n";
            return 0;
        }
        if (argc != 11) throw std::runtime_error("Usage: runner SCORES OUTPUT ROWS COLS K DEVICE TOOLKIT WARMUP REPEATS RESERVED_MODE(resident)");
        const fs::path scoresPath = fs::canonical(argv[1]), outputDir = fs::canonical(argv[2]);
        const auto rows = Integer(argv[3], 1, INT32_MAX), cols = Integer(argv[4], 1, INT32_MAX);
        const auto k = Integer(argv[5], 1, std::min<int64_t>(32, cols));
        const int device = static_cast<int>(Integer(argv[6], 0, INT32_MAX));
        const fs::path toolkit = fs::canonical(argv[7]);
        const int warmup = static_cast<int>(Integer(argv[8], 1, 10000));
        const int repeats = static_cast<int>(Integer(argv[9], 1, 10000));
        if (std::string(argv[10]) != "resident" || rows > INT32_MAX/cols)
            throw std::runtime_error("Only bounded resident FP32 reference is supported");
        if (!fs::is_directory(outputDir) || !fs::is_empty(outputDir)) throw std::runtime_error("Output directory must exist and be empty");
        const auto inputBytes = Bytes(static_cast<uint64_t>(rows*cols), sizeof(float));
        const auto valueBytes = Bytes(static_cast<uint64_t>(rows*k), sizeof(float));
        const auto idBytes = Bytes(static_cast<uint64_t>(rows*k), sizeof(int64_t));
        Stage("validate_input");
        if (fs::file_size(scoresPath) != inputBytes) throw std::runtime_error("Score file size mismatch");
        std::vector<float> input(static_cast<size_t>(rows*cols));
        std::ifstream f(scoresPath, std::ios::binary);
        if (!f.read(reinterpret_cast<char *>(input.data()), static_cast<std::streamsize>(inputBytes))) throw std::runtime_error("Cannot read scores");
        for (float v : input) if (!std::isfinite(v) || v < 0) throw std::runtime_error("Input must be finite nonnegative FP32");
        Library library(toolkit/"lib64/libopapi.so");
        auto prepare = library.Symbol<Prepare>("aclnnTopkGetWorkspaceSize");
        auto execute = library.Symbol<Execute>("aclnnTopk");
        Runtime runtime;
        Stage("runtime_init");
        Require(aclInit(nullptr), "aclInit"); runtime.initialized = true;
        Require(aclrtSetDevice(device), "aclrtSetDevice"); runtime.device = device; runtime.selected = true;
        Require(aclrtCreateContext(&runtime.context, device), "aclrtCreateContext");
        Require(aclrtCreateStream(&runtime.stream), "aclrtCreateStream");
        Resources resources; resources.stream = runtime.stream;
        void *x = resources.Allocate(inputBytes), *y = resources.Allocate(valueBytes), *ids = resources.Allocate(idBytes);
        auto *xTensor = resources.Tensor(rows, cols, ACL_FLOAT, x);
        auto *yTensor = resources.Tensor(rows, k, ACL_FLOAT, y);
        auto *idTensor = resources.Tensor(rows, k, ACL_INT64, ids);
        Stage("upload_once");
        Require(aclrtMemcpy(x, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload");
        uint64_t capacity = 0;
        void *workspace = nullptr;
        struct Sample { double prepare, launch, batch; };
        auto launch = [&](bool bootstrap) {
            uint64_t needed = 0; aclOpExecutor *executor = nullptr;
            const auto a = Clock::now();
            Require(prepare(xTensor, k, -1, true, true, yTensor, idTensor, &needed, &executor), "aclnnTopkGetWorkspaceSize");
            if (!executor) throw std::runtime_error("Official executor is null");
            const auto b = Clock::now();
            if (bootstrap) {
                if (needed > std::numeric_limits<size_t>::max()) throw std::runtime_error("Workspace too large");
                capacity = needed; workspace = resources.Allocate(static_cast<size_t>(needed));
                std::cout << "workspace_capacity_bytes=" << capacity << std::endl;
            } else if (needed > capacity) throw std::runtime_error("Official workspace grew; stop instead of allocating in timed loop");
            const auto c = Clock::now();
            Require(execute(workspace, needed, executor, runtime.stream), "aclnnTopk");
            Require(aclrtSynchronizeStream(runtime.stream), "synchronize");
            const auto d = Clock::now();
            return Sample{Ms(a,b), Ms(c,d), Ms(a,d)};
        };
        auto poison = [&]() {
            Require(aclrtMemset(y, valueBytes, 255, valueBytes), "poison values");
            Require(aclrtMemset(ids, idBytes, 255, idBytes), "poison ids");
        };
        auto download = [&]() {
            Output o{std::vector<float>(static_cast<size_t>(rows*k)), std::vector<int64_t>(static_cast<size_t>(rows*k))};
            Require(aclrtMemcpy(o.values.data(), valueBytes, y, valueBytes, ACL_MEMCPY_DEVICE_TO_HOST), "download values");
            Require(aclrtMemcpy(o.ids.data(), idBytes, ids, idBytes, ACL_MEMCPY_DEVICE_TO_HOST), "download ids");
            Validate(input, o, rows, cols, k); return o;
        };
        Stage("bootstrap_and_validate");
        poison(); launch(true); const Output first = download();
        Stage("poisoned_warmup_repeat_checks");
        for (int i = 0; i < warmup; ++i) {
            poison(); launch(false);
            if (!Equal(first, download())) throw std::runtime_error("Official repeated output changed");
        }
        launch(false); // One plain warmup after diagnostic memory operations.
        Stage("measure_resident_topk");
        std::vector<Sample> samples;
        for (int i = 0; i < repeats; ++i) samples.push_back(launch(false));
        const Output measured = download();
        if (!Equal(first, measured)) throw std::runtime_error("Last measured output changed");
        Stage("final_poisoned_repeat_check");
        poison(); launch(false); const Output final = download();
        if (!Equal(first, final)) throw std::runtime_error("Final poisoned output changed");
        Stage("release_resources");
        if (!resources.Close() || !runtime.Close()) throw std::runtime_error("Cleanup failed");
        Stage("save_device_output");
        Write(outputDir/"bootstrap_scores.bin", first.values); Write(outputDir/"bootstrap_ids.bin", first.ids);
        Write(outputDir/"top_scores.bin", final.values); Write(outputDir/"top_ids.bin", final.ids);
        std::vector<double> batch;
        for (const auto &s : samples) batch.push_back(s.batch);
        const double median = Median(batch);
        std::ofstream report(outputDir/"official_timing.json");
        report << std::setprecision(17)
               << "{\n\"status\":\"official_topk_calls_completed_checks_pending\",\n\"api\":\"aclnnTopk\","
               << "\n\"mode\":\"resident\",\n\"indices_dtype\":\"int64\",\n\"device\":" << device
               << ",\n\"rows\":" << rows << ",\n\"cols\":" << cols << ",\n\"k\":" << k
               << ",\n\"warmup\":" << warmup << ",\n\"repeats\":" << repeats
               << ",\n\"additional_plain_warmup\":1,\n\"validated_poisoned_calls\":" << warmup+2
               << ",\n\"verified_api_symbols\":2,\n\"input_upload_count\":1,\n\"workspace_capacity_bytes\":" << capacity
               << ",\n\"median_batch_path_ms\":" << median << ",\n\"samples\":[";
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto &s = samples[i];
            if (i) report << ',';
            report << "{\"prepare_ms\":" << s.prepare << ",\"launch_sync_ms\":" << s.launch << ",\"batch_path_ms\":" << s.batch << '}';
        }
        report << "]\n}\n"; report.close();
        if (!report) throw std::runtime_error("Cannot save official timing");
        std::cout << "official_topk_resident_median_batch_ms=" << std::setprecision(9) << median
                  << "\nFP32 official reference completed; independent full-ranking validation still required. No BM25 speedup claim.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAILED: " << e.what() << "; no CPU replacement output\n"; return 1;
    }
}
