// Actual generated Radix signatures confirmed from the user's installed A5 header.
// Correctness diagnostic only: all outputs come from device memory, never a CPU fallback.
#include "radix_host_common.h"
#include <algorithm>
#include <numeric>

std::vector<int32_t> FullSortIds(const std::vector<float> &input, int64_t q, int64_t docs, int64_t k)
{
    std::vector<int32_t> expected(static_cast<size_t>(q*k));
    std::vector<int32_t> row(static_cast<size_t>(docs));
    for (int64_t query = 0; query < q; ++query) {
        const size_t base = static_cast<size_t>(query*docs);
        std::iota(row.begin(), row.end(), 0);
        // Independent numerical full sort; does not use the device's integer radix key.
        std::sort(row.begin(), row.end(), [&](int32_t a, int32_t b) {
            return input[base+a] > input[base+b] || (input[base+a] == input[base+b] && a < b);
        });
        std::copy_n(row.begin(), static_cast<size_t>(k), expected.begin()+query*k);
    }
    return expected;
}

void CheckFullOutput(const std::vector<float> &input, const std::vector<int32_t> &expected,
                     const std::vector<float> &scores, const std::vector<int32_t> &ids,
                     int64_t docs, int64_t k)
{
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != expected[i])
            throw std::runtime_error("Full-sort ID mismatch at query=" + std::to_string(i/k) +
                                     " rank=" + std::to_string(i%k) + " actual=" + std::to_string(ids[i]) +
                                     " expected=" + std::to_string(expected[i]));
        const size_t source = (i/static_cast<size_t>(k))*static_cast<size_t>(docs)+static_cast<size_t>(ids[i]);
        if (std::memcmp(&scores[i], &input[source], sizeof(float)) != 0)
            throw std::runtime_error("Original FP32 score bits differ at record=" + std::to_string(i));
    }
}

int main(int argc, char **argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-library") {
            Library api(fs::absolute(argv[2]));
            api.Symbol<PrepareSignature>("aclnnBm25TopKGetWorkspaceSize");
            api.Symbol<ExecuteSignature>("aclnnBm25TopK");
            std::cout << "Both Radix API symbol owners verified. No device work submitted.\n";
            return 0;
        }
        if (argc != 9) {
            std::cerr << "Usage: bm25_topk_radix_runner INPUT OUTPUT Q D K DEVICE API_LIBRARY CALLS\n";
            return 2;
        }
        static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4 && std::numeric_limits<float>::is_iec559);
        const uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian) != 1) throw std::runtime_error("Little-endian host required");
        Stage("validate_inputs_and_build_full_sort_oracles");
        const int64_t q = ParseInteger(argv[3], "queries", 1, INT32_MAX);
        const int64_t docs = ParseInteger(argv[4], "docs", 1, INT32_MAX);
        const int64_t k = ParseInteger(argv[5], "K", 1, 32);
        const int64_t device = ParseInteger(argv[6], "device", 0, INT32_MAX);
        const int64_t calls = ParseInteger(argv[8], "calls", 5, 100);
        const uint64_t count = static_cast<uint64_t>(q)*docs;
        if (k > docs || count > INT32_MAX) throw std::runtime_error("Diagnostic dimensions exceed supported limits");
        const auto output = fs::absolute(argv[2]);
        if (fs::exists(output)) throw std::runtime_error("Choose a fresh output directory");
        const size_t inputBytes = CheckedBytes(count, sizeof(float));
        const size_t outCount = static_cast<size_t>(q*k);
        const size_t outputBytes = CheckedBytes(outCount, sizeof(float));
        std::vector<float> input[2];
        std::vector<int32_t> expected[2];
        for (int v = 0; v < 2; ++v) {
            input[v] = Read<float>(fs::absolute(argv[1])/(v == 0 ? "input_a" : "input_b")/"scores.bin");
            if (input[v].size() != count) throw std::runtime_error("Input shape mismatch");
            for (float value : input[v])
                if (!std::isfinite(value) || value < 0) throw std::runtime_error("Only nonnegative finite FP32 inputs accepted");
            expected[v] = FullSortIds(input[v], q, docs, k);
        }
        Library api(fs::absolute(argv[7]));
        const auto prepare = api.Symbol<PrepareSignature>("aclnnBm25TopKGetWorkspaceSize");
        const auto execute = api.Symbol<ExecuteSignature>("aclnnBm25TopK");
        Runtime runtime;
        runtime.device = static_cast<int>(device);
        Stage("runtime_init");
        Require(aclInit(nullptr), "aclInit"); runtime.initialized = true;
        Require(aclrtSetDevice(runtime.device), "aclrtSetDevice"); runtime.deviceSelected = true;
        Require(aclrtCreateContext(&runtime.context, runtime.device), "aclrtCreateContext");
        aclrtRunMode mode;
        Require(aclrtGetRunMode(&mode), "aclrtGetRunMode");
        if (mode != ACL_HOST) throw std::runtime_error("ACL_HOST mode required");
        Require(aclrtCreateStream(&runtime.stream), "aclrtCreateStream");
        Resources resources;
        resources.stream = runtime.stream;
        void *inputMemory = resources.Allocate(inputBytes);
        void *scoreMemory = resources.Allocate(outputBytes);
        void *idMemory = resources.Allocate(outputBytes);
        auto *inputTensor = resources.Describe(q, docs, ACL_FLOAT, inputMemory);
        auto *scoreTensor = resources.Describe(q, k, ACL_FLOAT, scoreMemory);
        auto *idTensor = resources.Describe(q, k, ACL_INT32, idMemory);
        void *workspace = nullptr;
        uint64_t capacity = 0;
        std::vector<float> scores(outCount), savedScores[2];
        std::vector<int32_t> ids(outCount), savedIds[2];
        int64_t seen[2] = {0, 0};
        for (int64_t call = 0; call < calls; ++call) {
            const int variant = static_cast<int>(call % 2);
            std::cout << "stage=upload_poison_prepare_execute_sync_download_check call=" << call
                      << " input=" << (variant == 0 ? "A" : "B") << std::endl;
            Require(aclrtMemcpy(inputMemory, inputBytes, input[variant].data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "input upload");
            std::fill(scores.begin(), scores.end(), std::numeric_limits<float>::quiet_NaN());
            std::fill(ids.begin(), ids.end(), -1);
            Require(aclrtMemcpy(scoreMemory, outputBytes, scores.data(), outputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "score poison");
            Require(aclrtMemcpy(idMemory, outputBytes, ids.data(), outputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "ID poison");
            uint64_t bytes = 0;
            aclOpExecutor *executor = nullptr;
            Require(prepare(inputTensor, k, scoreTensor, idTensor, &bytes, &executor), "Radix prepare");
            if (executor == nullptr) throw std::runtime_error("Radix prepare returned a null executor");
            if (call == 0) {
                capacity = bytes;
                workspace = resources.Allocate(CheckedBytes(bytes, 1));
                std::cout << "workspace_bytes=" << capacity << " declared_user_gm_scratch_bytes=0" << std::endl;
            } else if (bytes != capacity) {
                throw std::runtime_error("Workspace size changed between same-shape calls");
            }
            Require(execute(workspace, bytes, executor, runtime.stream), "Radix execute");
            executor = nullptr; // fresh ordinary executor consumed during this call
            Require(aclrtSynchronizeStream(runtime.stream), "Radix synchronize");
            Require(aclrtMemcpy(scores.data(), outputBytes, scoreMemory, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), "score download");
            Require(aclrtMemcpy(ids.data(), outputBytes, idMemory, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), "ID download");
            CheckFullOutput(input[variant], expected[variant], scores, ids, docs, k);
            if (seen[variant] == 0) { savedScores[variant] = scores; savedIds[variant] = ids; }
            else if (std::memcmp(scores.data(), savedScores[variant].data(), outputBytes) != 0 ||
                     std::memcmp(ids.data(), savedIds[variant].data(), outputBytes) != 0)
                throw std::runtime_error("Repeated input output bytes differ after alternating inputs");
            ++seen[variant];
            std::cout << "full_sort_and_original_bits_passed call=" << call << std::endl;
        }
        Stage("release_resources");
        if (!resources.Close() || !runtime.Close()) throw std::runtime_error("Cleanup failed; no success outputs saved");
        Stage("save_actual_device_outputs");
        fs::create_directories(output.parent_path());
        if (!fs::create_directory(output)) throw std::runtime_error("Output directory creation failed");
        for (int v = 0; v < 2; ++v) {
            const std::string prefix = v == 0 ? "a_" : "b_";
            Write(output/(prefix+"top_scores.bin"), savedScores[v]);
            Write(output/(prefix+"top_ids.bin"), savedIds[v]);
        }
        std::ofstream record(output/"device_validation.json");
        record << "{\"status\":\"radix_device_calls_and_cpp_full_sort_passed\",\"queries\":" << q
               << ",\"docs\":" << docs << ",\"k\":" << k << ",\"device\":" << device
               << ",\"validated_calls\":" << calls << ",\"workspace_bytes\":" << capacity
               << ",\"declared_user_gm_scratch_bytes\":0,\"declared_ub_algorithm_bytes\":8512"
               << ",\"input_calls\":[" << seen[0] << "," << seen[1]
               << "],\"every_call_poisoned\":true,\"every_call_full_sort_checked\":true}\n";
        record.close();
        if (!record) throw std::runtime_error("Cannot save device validation record");
        std::cout << "Radix device calls and every-call full-sort/bit checks completed. Python independent check still required. No performance measurement.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
