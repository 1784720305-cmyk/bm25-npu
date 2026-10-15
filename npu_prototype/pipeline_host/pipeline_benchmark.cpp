#include "pipeline_common.h"
#include <algorithm>
#include <iomanip>
#include <cstdlib>

struct Sample {
    double path, prepare, launchSync;
    double contributionSync = 0, accumulateSync = 0, topkSync = 0;
};

template<class T> bool EqualBytes(const std::vector<T> &a, const std::vector<T> &b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

double Percentile(const std::vector<Sample> &samples, double fraction, int field)
{
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto &sample : samples) {
        const double fields[] = {sample.path, sample.prepare, sample.launchSync,
                                 sample.contributionSync, sample.accumulateSync, sample.topkSync};
        if (field < 0 || field >= 6) throw std::runtime_error("Invalid timing field");
        values.push_back(fields[field]);
    }
    std::sort(values.begin(), values.end());
    const size_t rank = std::max<size_t>(1, static_cast<size_t>(std::ceil(fraction * values.size())));
    return values.at(rank - 1);
}

void SaveTiming(const fs::path &path, const std::vector<Sample> &samples, const Sample &cold,
                const std::string &mode, int64_t device, int64_t queries, int64_t docs, int64_t k,
                int64_t warmup, int64_t repeats, const uint64_t (&capacity)[3])
{
    std::ofstream file(path);
    file << std::setprecision(17)
         << "{\n  \"status\": \"device_repeated_completed_independent_correctness_not_yet_checked\",\n"
         << "  \"mode\": \"" << mode << "\",\n"
         << "  \"scope\": \"prepared input; includes fresh three-stage API preparation, launch and stream synchronization; workspace/buffers allocated before measurement; excludes CPU input packing, initialization, file IO and debug downloads\",\n"
         << "  \"transfer_policy\": \"" << (mode != "with_transfers" ? "four inputs uploaded once; Top-K download excluded" : "four inputs uploaded and Top-K scores/IDs downloaded each iteration") << "\",\n"
         << "  \"synchronization_policy\": \"" << (mode == "profile_stages" ? "synchronize after EACH operator; diagnostic only, not normal pipeline latency or pure kernel time" : "synchronize after the entire three-operator chain") << "\",\n"
         << "  \"executor_policy\": \"fresh ordinary executors every iteration, framework-owned; three workspaces reused only after synchronization\",\n"
         << "  \"percentile_method\": \"nearest_rank\",\n"
         << "  \"full_repeat_validation\": \"every specified warmup and final measured iteration, all four arrays byte-identical to bootstrap; outputs poisoned before these checks\",\n"
         << "  \"last_measured_iteration_poisoned_outside_timer\": true,\n"
         << "  \"all_measured_topk_checked\": " << (mode == "with_transfers" ? "true" : "false") << ",\n"
         << "  \"device\": " << device << ",\n  \"queries\": " << queries << ",\n  \"docs\": " << docs << ",\n  \"k\": " << k << ",\n"
         << "  \"warmup\": " << warmup << ",\n  \"additional_plain_warmup\": 1,\n  \"repeats\": " << repeats << ",\n"
         << "  \"workspace_capacity_bytes\": [" << capacity[0] << ", " << capacity[1] << ", " << capacity[2] << "],\n"
         << "  \"bootstrap_batch_path_ms\": " << cold.path << ",\n"
         << "  \"median_batch_path_ms\": " << Percentile(samples, .5, 0) << ",\n"
         << "  \"p95_batch_path_ms\": " << Percentile(samples, .95, 0) << ",\n"
         << "  \"median_prepare_ms\": " << Percentile(samples, .5, 1) << ",\n"
         << "  \"median_launch_sync_ms\": " << Percentile(samples, .5, 2) << ",\n";
    if (mode == "profile_stages") {
        file << "  \"median_contribution_launch_sync_ms\": " << Percentile(samples, .5, 3) << ",\n"
             << "  \"median_accumulate_launch_sync_ms\": " << Percentile(samples, .5, 4) << ",\n"
             << "  \"median_topk_launch_sync_ms\": " << Percentile(samples, .5, 5) << ",\n";
    }
    file << "  \"samples\": [\n";
    for (size_t i = 0; i < samples.size(); ++i) {
        file << (i ? ",\n" : "") << "    {\"batch_path_ms\": " << samples[i].path
             << ", \"prepare_ms\": " << samples[i].prepare << ", \"launch_sync_ms\": " << samples[i].launchSync;
        if (mode == "profile_stages")
            file << ", \"contribution_launch_sync_ms\": " << samples[i].contributionSync
                 << ", \"accumulate_launch_sync_ms\": " << samples[i].accumulateSync
                 << ", \"topk_launch_sync_ms\": " << samples[i].topkSync;
        file << "}";
    }
    file << "\n  ]\n}\n";
    file.close();
    if (!file) throw std::runtime_error("Cannot save NPU timing JSON");
}

int main(int argc, char **argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-libraries") {
            Interfaces api(fs::absolute(argv[2]));
            std::cout << "All six generated API symbols belong to the requested libraries. No device work submitted.\n";
            return 0;
        }
        if (argc != 11) {
            std::cerr << "Usage: bm25_pipeline_benchmark INPUT_DIR OUTPUT_DIR QUERIES DOCS K DEVICE VENDOR_DIR WARMUP REPEATS MODE\n"
                      << "       bm25_pipeline_benchmark --check-libraries VENDOR_DIR\n";
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
        const int64_t warmup = ParseInteger(argv[8], "warmup", 1, 10000);
        const int64_t repeats = ParseInteger(argv[9], "repeats", 1, 10000);
        const std::string timingMode(argv[10]);
        const char *scratchEnv = std::getenv("BM25_EXPECT_TOPK_SCRATCH_BYTES");
        const uint64_t expectedTopkScratch = scratchEnv == nullptr ? 0 :
            static_cast<uint64_t>(ParseInteger(scratchEnv, "expected Top-K scratch", 1, INT64_MAX));
        if (timingMode != "resident" && timingMode != "with_transfers" && timingMode != "profile_stages")
            throw std::runtime_error("MODE must be resident, with_transfers or profile_stages");
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

        const std::vector<float> *hostInputs[3] = {&tf, &norm, &idf};
        auto upload = [&]() {
            for (int i = 0; i < 3; ++i)
                Require(aclrtMemcpy(inputs[i], valueBytes, hostInputs[i]->data(), valueBytes, ACL_MEMCPY_HOST_TO_DEVICE), "input upload");
            Require(aclrtMemcpy(offsetMemory, offsetBytes, offsets.data(), offsetBytes, ACL_MEMCPY_HOST_TO_DEVICE), "offset upload");
        };
        auto downloadTop = [&]() {
            Require(aclrtMemcpy(actualTop.data(), topBytes, topScoreMemory, topBytes, ACL_MEMCPY_DEVICE_TO_HOST), "top score download");
            Require(aclrtMemcpy(actualIds.data(), topBytes, topIdMemory, topBytes, ACL_MEMCPY_DEVICE_TO_HOST), "top ID download");
        };
        auto downloadDebug = [&]() {
            downloadTop();
            Require(aclrtMemcpy(actualContribution.data(), valueBytes, contributionMemory, valueBytes, ACL_MEMCPY_DEVICE_TO_HOST), "debug contribution download");
            Require(aclrtMemcpy(actualScores.data(), scoreBytes, scoreMemory, scoreBytes, ACL_MEMCPY_DEVICE_TO_HOST), "debug full score download");
            for (float value : actualContribution)
                if (!std::isfinite(value) || value < 0) throw std::runtime_error("Invalid device contribution");
            for (float value : actualScores)
                if (!std::isfinite(value) || value < 0) throw std::runtime_error("Invalid device score");
            ValidateOutput(actualScores, actualTop, actualIds, queries, docs, k);
        };
        // The framework consumes normal executors. Obtain new ones on EVERY iteration.
        // Keep three separate workspace allocations until the final stream synchronization.
        uint64_t capacity[3] = {0, 0, 0};
        void *workspace[3] = {nullptr, nullptr, nullptr};
        auto iteration = [&](bool allocate) {
            const auto begin = std::chrono::steady_clock::now();
            if (timingMode == "with_transfers") upload();
            const auto prepareStart = std::chrono::steady_clock::now();
            uint64_t size[3] = {0, 0, 0};
            aclOpExecutor *executor[3] = {nullptr, nullptr, nullptr};
            Require(api.prepareContribution(inputTensors[0], inputTensors[1], inputTensors[2], contributionTensor,
                                            &size[0], &executor[0]), "contribution prepare");
            Require(api.prepareAccumulate(contributionTensor, offsetTensor, flatScoreTensor,
                                          &size[1], &executor[1]), "accumulate prepare");
            Require(api.prepareTopK(matrixScoreTensor, k, topScoreTensor, topIdTensor,
                                   &size[2], &executor[2]), "Top-K prepare");
            if (expectedTopkScratch != 0 && (size[0] > UINT64_MAX - expectedTopkScratch ||
                size[2] < size[0] + expectedTopkScratch))
                throw std::runtime_error("Parallel Top-K workspace mismatch; expected new tiling plus scratch, stop before execution");
            for (int i = 0; i < 3; ++i) {
                if (executor[i] == nullptr || size[i] > std::numeric_limits<size_t>::max())
                    throw std::runtime_error("Invalid executor/workspace");
                if (allocate) {
                    capacity[i] = size[i];
                    workspace[i] = resources.Allocate(static_cast<size_t>(size[i]));
                    std::cout << "workspace_stage=" << i << " capacity_bytes=" << capacity[i] << std::endl;
                } else if (size[i] > capacity[i]) {
                    throw std::runtime_error("Workspace grew; stop instead of allocating inside the measured path");
                }
            }
            const auto launchStart = std::chrono::steady_clock::now();
            double contributionSync = 0, accumulateSync = 0, topkSync = 0;
            auto stageEnd = launchStart;
            Require(api.executeContribution(workspace[0], size[0], executor[0], runtime.stream), "contribution execute"); executor[0] = nullptr;
            if (timingMode == "profile_stages") {
                Require(aclrtSynchronizeStream(runtime.stream), "profile contribution synchronize");
                stageEnd = std::chrono::steady_clock::now();
                contributionSync = std::chrono::duration<double, std::milli>(stageEnd - launchStart).count();
            }
            const auto accumulateStart = stageEnd;
            Require(api.executeAccumulate(workspace[1], size[1], executor[1], runtime.stream), "accumulate execute"); executor[1] = nullptr;
            if (timingMode == "profile_stages") {
                Require(aclrtSynchronizeStream(runtime.stream), "profile accumulate synchronize");
                stageEnd = std::chrono::steady_clock::now();
                accumulateSync = std::chrono::duration<double, std::milli>(stageEnd - accumulateStart).count();
            }
            const auto topkStart = stageEnd;
            Require(api.executeTopK(workspace[2], size[2], executor[2], runtime.stream), "Top-K execute"); executor[2] = nullptr;
            Require(aclrtSynchronizeStream(runtime.stream), "pipeline synchronize");
            const auto launchEnd = std::chrono::steady_clock::now();
            if (timingMode == "profile_stages")
                topkSync = std::chrono::duration<double, std::milli>(launchEnd - topkStart).count();
            if (timingMode == "with_transfers") downloadTop();
            const auto end = std::chrono::steady_clock::now();
            // Return host wall times. launch_sync is NOT pure device-kernel time.
            return Sample{
                std::chrono::duration<double, std::milli>(end - begin).count(),
                std::chrono::duration<double, std::milli>(launchStart - prepareStart).count(),
                std::chrono::duration<double, std::milli>(launchEnd - launchStart).count(),
                contributionSync, accumulateSync, topkSync};
        };
        Stage("bootstrap_upload_prepare_execute");
        if (timingMode != "with_transfers") upload();
        const auto cold = iteration(true);
        Stage("bootstrap_debug_download");
        downloadDebug();
        const auto baselineContribution = actualContribution;
        const auto baselineScores = actualScores;
        const auto baselineTop = actualTop;
        const auto baselineIds = actualIds;
        auto requireSameTop = [&]() {
            if (!EqualBytes(actualTop, baselineTop) || !EqualBytes(actualIds, baselineIds))
                throw std::runtime_error("Repeated Top-K bytes changed");
        };
        auto requireSameAll = [&]() {
            requireSameTop();
            if (!EqualBytes(actualContribution, baselineContribution) || !EqualBytes(actualScores, baselineScores))
                throw std::runtime_error("Repeated contribution/score bytes changed");
        };
        // 0xffffffff is NaN for IEEE FP32 and -1 for int32: overwritten outputs are required.
        std::vector<uint32_t> poison(std::max({tf.size(), static_cast<size_t>(groupCount), static_cast<size_t>(topCount)}), UINT32_MAX);
        auto poisonOutputs = [&]() {
            Require(aclrtMemcpy(contributionMemory, valueBytes, poison.data(), valueBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison contribution");
            Require(aclrtMemcpy(scoreMemory, scoreBytes, poison.data(), scoreBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison scores");
            Require(aclrtMemcpy(topScoreMemory, topBytes, poison.data(), topBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison top scores");
            Require(aclrtMemcpy(topIdMemory, topBytes, poison.data(), topBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison IDs");
        };
        Stage("warmup_and_full_repeat_validation");
        for (int64_t i = 0; i < warmup; ++i) {
            poisonOutputs();
            iteration(false);
            downloadDebug();
            requireSameAll();
        }
        // A final ordinary warmup removes the diagnostic downloads from the measurement boundary.
        iteration(false);
        Stage("measure_repeated_pipeline");
        std::vector<Sample> samples;
        samples.reserve(static_cast<size_t>(repeats));
        for (int64_t i = 0; i < repeats; ++i) {
            // Untimed last-round poison makes stale output fail final full-array validation.
            if (i == repeats - 1) poisonOutputs();
            const auto sample = iteration(false);
            samples.push_back(sample);
            if (timingMode == "with_transfers") requireSameTop(); // Outside the measured interval.
        }
        Stage("final_debug_download_and_repeat_check");
        downloadDebug();
        requireSameAll();
        Stage("release_resources");
        if (!resources.Close() || !runtime.Close()) throw std::runtime_error("Cleanup failed; results not saved");
        Stage("save_output");
        fs::create_directories(outputDir.parent_path());
        if (!fs::create_directory(outputDir)) throw std::runtime_error("Choose a fresh output directory");
        Write(outputDir / "contributions.bin", actualContribution);
        Write(outputDir / "scores.bin", actualScores);
        Write(outputDir / "top_scores.bin", actualTop);
        Write(outputDir / "top_ids.bin", actualIds);
        fs::create_directory(outputDir / "bootstrap");
        Write(outputDir / "bootstrap/contributions.bin", baselineContribution);
        Write(outputDir / "bootstrap/scores.bin", baselineScores);
        Write(outputDir / "bootstrap/top_scores.bin", baselineTop);
        Write(outputDir / "bootstrap/top_ids.bin", baselineIds);
        SaveTiming(outputDir / "npu_timing.json", samples, cold, timingMode, device, queries, docs, k,
                   warmup, repeats, capacity);
        std::cout << "device=" << device << " mode=" << timingMode << " measured_batches=" << repeats
                  << " validated_full_repeat_batches=" << warmup + 1 << " intermediate_host_roundtrips=0\n"
                  << "median_batch_path_ms=" << Percentile(samples, .5, 0)
                  << "\nmedian_prepare_ms=" << Percentile(samples, .5, 1)
                  << "\nmedian_three_launches_and_sync_ms=" << Percentile(samples, .5, 2) << '\n';
        if (timingMode == "profile_stages") {
            std::cout << "median_contribution_launch_sync_ms=" << Percentile(samples, .5, 3)
                      << "\nmedian_accumulate_launch_sync_ms=" << Percentile(samples, .5, 4)
                      << "\nmedian_topk_launch_sync_ms=" << Percentile(samples, .5, 5)
                      << "\nPer-stage synchronization changes the normal pipeline; diagnostic only, not pure kernel time.\n";
        }
        std::cout << "Repeated call completed; check independent CPU formula and all-query cross-device ranking. "
                  << "Prepared-input experiment, not complete retrieval or a speedup claim.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
