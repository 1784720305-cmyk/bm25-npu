#include "pipeline_common.h"

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
