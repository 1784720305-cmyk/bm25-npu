// Warp signatures confirmed from the user's installed A5 header on 2026-10-16.
// Reuse runtime, tensor and absolute-library helpers; never instantiate the three-stage Interfaces.
#include "../pipeline_host/pipeline_common.h"
#include <algorithm>

int main(int argc, char **argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-library") {
            Library api(fs::absolute(argv[2]));
            api.Symbol<PrepareSignature>("aclnnBm25TopKGetWorkspaceSize");
            api.Symbol<ExecuteSignature>("aclnnBm25TopK");
            std::cout << "Both warp API symbol owners verified. No device work submitted.\n";
            return 0;
        }
        if (argc != 11) {
            std::cerr << "Usage: bm25_topk_warp_runner INPUT OUTPUT Q D K DEVICE API_LIBRARY SCRATCH_BYTES SYSTEM_BYTES CALLS\n";
            return 2;
        }
        static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4);
        static_assert(std::numeric_limits<float>::is_iec559);
        const uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian) != 1) throw std::runtime_error("Little-endian host required");
        Stage("validate_inputs");
        const int64_t q = ParseInteger(argv[3], "queries", 1, INT32_MAX);
        const int64_t docs = ParseInteger(argv[4], "docs", 1, INT32_MAX);
        const int64_t k = ParseInteger(argv[5], "K", 1, 32);
        const int64_t device = ParseInteger(argv[6], "device", 0, INT32_MAX);
        const uint64_t expectedScratch = ParseInteger(argv[8], "scratch bytes", 1, INT64_MAX);
        const uint64_t systemBytes = ParseInteger(argv[9], "system bytes", 1, INT64_MAX);
        const int64_t calls = ParseInteger(argv[10], "calls", 2, 100);
        const uint64_t inputCount = static_cast<uint64_t>(q) * docs;
        if (k > docs || inputCount > INT32_MAX) throw std::runtime_error("K or diagnostic input size exceeds limits");
        uint64_t partitions = 32;
        while (partitions < 256 && static_cast<uint64_t>(docs) > partitions * 16) partitions *= 2;
        const uint64_t scratch = static_cast<uint64_t>(q) * partitions * 12;
        if (expectedScratch != scratch || systemBytes > UINT64_MAX - scratch)
            throw std::runtime_error("Warp workspace contract mismatch");
        const uint64_t expectedWorkspace = systemBytes + scratch;
        CheckedBytes(expectedWorkspace, 1);
        const auto output = fs::absolute(argv[2]);
        if (fs::exists(output)) throw std::runtime_error("Choose a fresh output directory");
        const size_t inputBytes = CheckedBytes(inputCount, sizeof(float));
        const size_t outCount = static_cast<size_t>(q * k);
        const size_t outputBytes = CheckedBytes(outCount, sizeof(float));
        const auto input = Read<float>(fs::absolute(argv[1]) / "scores.bin");
        if (input.size() != inputCount) throw std::runtime_error("Input shape mismatch");
        for (float value : input) if (!std::isfinite(value) || value < 0) throw std::runtime_error("Invalid score input");
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
        Stage("upload_input_once");
        Require(aclrtMemcpy(inputMemory, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "input upload");
        void *workspace = nullptr;
        std::vector<float> scores(outCount), firstScores;
        std::vector<int32_t> ids(outCount), firstIds;
        for (int64_t call = 0; call < calls; ++call) {
            std::cout << "stage=poison_prepare_execute_sync_download call=" << call << std::endl;
            // Poison BOTH outputs for EVERY call; a missing write cannot pass as an old valid result.
            std::fill(scores.begin(), scores.end(), std::numeric_limits<float>::quiet_NaN());
            std::fill(ids.begin(), ids.end(), -1);
            Require(aclrtMemcpy(scoreMemory, outputBytes, scores.data(), outputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "score poison");
            Require(aclrtMemcpy(idMemory, outputBytes, ids.data(), outputBytes, ACL_MEMCPY_HOST_TO_DEVICE), "ID poison");
            uint64_t bytes = 0;
            aclOpExecutor *executor = nullptr;
            Require(prepare(inputTensor, k, scoreTensor, idTensor, &bytes, &executor), "warp prepare");
            if (!executor || bytes != expectedWorkspace)
                throw std::runtime_error("Warp prepare workspace mismatch: actual=" + std::to_string(bytes) +
                                         " expected=" + std::to_string(expectedWorkspace));
            if (call == 0) {
                workspace = resources.Allocate(static_cast<size_t>(bytes));
                std::cout << "workspace_bytes=" << bytes << " user_scratch_bytes=" << scratch << std::endl;
            }
            // A fresh ordinary executor each call; framework consumes it during execution.
            Require(execute(workspace, bytes, executor, runtime.stream), "warp execute"); executor = nullptr;
            Require(aclrtSynchronizeStream(runtime.stream), "warp synchronize");
            Require(aclrtMemcpy(scores.data(), outputBytes, scoreMemory, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), "score download");
            Require(aclrtMemcpy(ids.data(), outputBytes, idMemory, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), "ID download");
            ValidateOutput(input, scores, ids, q, docs, k);
            if (call == 0) { firstScores = scores; firstIds = ids; }
            else if (std::memcmp(scores.data(), firstScores.data(), outputBytes) != 0 ||
                     std::memcmp(ids.data(), firstIds.data(), outputBytes) != 0)
                throw std::runtime_error("Same-process repeated outputs differ");
        }
        Stage("release_resources");
        if (!resources.Close() || !runtime.Close()) throw std::runtime_error("Cleanup failed; output not saved");
        Stage("save_output");
        fs::create_directories(output.parent_path());
        if (!fs::create_directory(output)) throw std::runtime_error("Output directory creation failed");
        Write(output / "top_scores.bin", scores);
        Write(output / "top_ids.bin", ids);
        std::ofstream record(output / "device_validation.json");
        record << "{\"status\":\"device_calls_completed_full_sort_not_yet_checked\",\"queries\":" << q
               << ",\"docs\":" << docs << ",\"k\":" << k << ",\"device\":" << device
               << ",\"validated_calls\":" << calls << ",\"workspace_bytes\":" << expectedWorkspace
               << ",\"user_scratch_bytes\":" << scratch << ",\"system_bytes\":" << systemBytes
               << ",\"every_call_poisoned\":true}\n";
        record.close();
        if (!record) throw std::runtime_error("Cannot save device validation record");
        std::cout << "Warp device calls completed; every call checked and byte-identical. Full-sort oracle still required. No performance measurement.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
