#include "../pipeline_host/pipeline_common.h"
#include "aclnn_bm25_top_k_merge.h"
#include <algorithm>
#include <iomanip>
#include <cstdlib>

struct Sample {
    double path, prepare, launchSync;

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
        const double fields[] = {sample.path, sample.prepare, sample.launchSync};
        if (field < 0 || field >= 3) throw std::runtime_error("Invalid timing field");
        values.push_back(fields[field]);
    }
    std::sort(values.begin(), values.end());
    const size_t rank = std::max<size_t>(1, static_cast<size_t>(std::ceil(fraction * values.size())));
    return values.at(rank - 1);
}

void SaveTiming(const fs::path &path, const std::vector<Sample> &samples, const Sample &cold,
                const std::string &mode, int64_t device, int64_t queries, int64_t docs, int64_t k,
                int64_t warmup, int64_t repeats, int64_t shards, const uint64_t (&capacity)[4],
                const std::string &topkImplementation, uint64_t scratch)
{
    std::ofstream file(path);
    file << std::setprecision(17)
         << "{\n  \"status\": \"device_repeated_completed_independent_correctness_not_yet_checked\",\n"
         << "  \"mode\": \"" << mode << "\",\n"
         << "  \"scope\": \"prepared input; includes fresh four-stage API preparation, launch and stream synchronization; workspace/buffers allocated before measurement; excludes CPU input packing, initialization, file IO and debug downloads\",\n"
         << "  \"transfer_policy\": \"" << (mode != "with_transfers" ? "four inputs uploaded once; Top-K download excluded" : "four inputs uploaded and Top-K scores/IDs downloaded each iteration") << "\",\n"
         << "  \"synchronization_policy\": \"synchronize after the entire four-operator chain\",\n"
         << "  \"executor_policy\": \"fresh ordinary executors every iteration, framework-owned; four workspaces reused only after synchronization\",\n"
         << "  \"percentile_method\": \"nearest_rank\",\n"
         << "  \"full_repeat_validation\": \"every specified warmup and final measured iteration, all six arrays byte-identical to bootstrap; outputs poisoned before these checks\",\n"
         << "  \"last_measured_iteration_poisoned_outside_timer\": true,\n"
         << "  \"all_measured_topk_checked\": " << (mode == "with_transfers" ? "true" : "false") << ",\n"
         << "  \"device\": " << device << ",\n  \"queries\": " << queries << ",\n  \"docs\": " << docs << ",\n  \"k\": " << k << ",\n"
         << "  \"shards\": " << shards << ",\n  \"pipeline_stages\": 4,\n"
         << "  \"topk_implementation\": \"" << topkImplementation << "\",\n"
         << "  \"local_topk_user_scratch_bytes\": " << scratch << ",\n"
         << "  \"warmup\": " << warmup << ",\n  \"additional_plain_warmup\": 1,\n  \"repeats\": " << repeats << ",\n"
         << "  \"workspace_capacity_bytes\": [" << capacity[0] << ", " << capacity[1] << ", " << capacity[2] << ", " << capacity[3] << "],\n"
         << "  \"bootstrap_batch_path_ms\": " << cold.path << ",\n"
         << "  \"median_batch_path_ms\": " << Percentile(samples, .5, 0) << ",\n"
         << "  \"p95_batch_path_ms\": " << Percentile(samples, .95, 0) << ",\n"
         << "  \"median_prepare_ms\": " << Percentile(samples, .5, 1) << ",\n"
         << "  \"median_launch_sync_ms\": " << Percentile(samples, .5, 2) << ",\n";

    file << "  \"samples\": [\n";
    for (size_t i = 0; i < samples.size(); ++i) {
        file << (i ? ",\n" : "") << "    {\"batch_path_ms\": " << samples[i].path
             << ", \"prepare_ms\": " << samples[i].prepare << ", \"launch_sync_ms\": " << samples[i].launchSync;
        file << "}";
    }
    file << "\n  ]\n}\n";
    file.close();
    if (!file) throw std::runtime_error("Cannot save NPU timing JSON");
}


// Confirmed from the actual installed header, subsequently compiled and unit-tested on A5.
using MergePrepare = aclnnStatus (*)(const aclTensor *, const aclTensor *, int64_t, int64_t,
                                     const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
static_assert(std::is_same<decltype(&aclnnBm25TopKMergeGetWorkspaceSize), MergePrepare>::value);
static_assert(std::is_same<decltype(&aclnnBm25TopKMerge), ExecuteSignature>::value);

struct ShardedInterfaces : Interfaces {
    Library merge;
    MergePrepare prepareMerge;
    ExecuteSignature executeMerge;
    explicit ShardedInterfaces(const fs::path &vendors)
        : Interfaces(vendors), merge(vendors / "bm25_topk_merge/op_api/lib/libcust_opapi.so")
    {
        prepareMerge = merge.Symbol<MergePrepare>("aclnnBm25TopKMergeGetWorkspaceSize");
        executeMerge = merge.Symbol<ExecuteSignature>("aclnnBm25TopKMerge");
    }
};

aclTensor *Describe3D(Resources &resources, int64_t q, int64_t shards, int64_t k, aclDataType type, void *memory)
{
    const int64_t dimensions[3] = {q, shards, k}, strides[3] = {shards*k, k, 1};
    auto *tensor = aclCreateTensor(dimensions, 3, type, strides, 0, ACL_FORMAT_ND, dimensions, 3, memory);
    if (!tensor) throw std::runtime_error("Cannot create 3D candidate description");
    resources.tensors.push_back(tensor);
    return tensor;
}

int main(int argc, char **argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-libraries") {
            ShardedInterfaces api(fs::absolute(argv[2]));
            std::cout << "All eight generated API symbols belong to the requested libraries. No device work submitted.\n";
            return 0;
        }
        if (argc != 12) {
            std::cerr << "Usage: bm25_sharded_benchmark INPUT_DIR OUTPUT_DIR QUERIES DOCS K DEVICE VENDOR_DIR WARMUP REPEATS MODE SHARDS\n"
                      << "       bm25_sharded_benchmark --check-libraries VENDOR_DIR\n";
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
        const int64_t shards = ParseInteger(argv[11], "shards", 1, 32);
        if (docs % shards != 0 || docs/shards < k) throw std::runtime_error("Documents must divide evenly; each shard needs K documents");
        const char *scratchEnv = std::getenv("BM25_EXPECT_TOPK_SCRATCH_BYTES");
        const uint64_t expectedTopkScratch = scratchEnv == nullptr ? 0 :
            static_cast<uint64_t>(ParseInteger(scratchEnv, "expected Top-K scratch", 0, INT64_MAX));
        const char *implementationEnv = std::getenv("BM25_TOPK_IMPLEMENTATION");
        const std::string topkImplementation = implementationEnv ? implementationEnv : "parallel";
        if (topkImplementation != "parallel" && topkImplementation != "warp" && topkImplementation != "radix")
            throw std::runtime_error("Unknown explicit Top-K implementation");
        const char *radixWorkspaceEnv = std::getenv("BM25_EXPECT_RADIX_WORKSPACE_BYTES");
        if (topkImplementation == "radix" && (scratchEnv == nullptr || radixWorkspaceEnv == nullptr))
            throw std::runtime_error("Radix requires explicit zero scratch and accepted unit workspace guards");
        const uint64_t expectedRadixWorkspace = topkImplementation == "radix" ?
            static_cast<uint64_t>(ParseInteger(radixWorkspaceEnv, "accepted Radix workspace", 0, INT64_MAX)) : 0;
        if (timingMode != "resident" && timingMode != "with_transfers")
            throw std::runtime_error("MODE must be resident or with_transfers");
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
        const uint64_t partialCount = topCount * shards;
        const size_t partialBytes = CheckedBytes(partialCount, sizeof(float));
        uint64_t partitions = 32;
        while (partitions < 256 && static_cast<uint64_t>(docs/shards) > partitions*16) partitions *= 2;
        const uint64_t scratch = topkImplementation == "radix" ? 0 :
            static_cast<uint64_t>(queries)*shards*partitions*(topkImplementation == "warp" ? 12 : (8*k+4));
        if (expectedTopkScratch != scratch) throw std::runtime_error("Explicit local Top-K scratch guard missing or inconsistent");
        std::cout << "value_count=" << tf.size() << " group_count=" << groupCount << " queries=" << queries
                  << " docs=" << docs << " k=" << k << " shards=" << shards << " pipeline_stages=4" << std::endl;
        std::cout << "topk_implementation=" << topkImplementation << " local_topk_user_scratch_bytes=" << scratch << std::endl;
        Stage("load_four_api_libraries");
        ShardedInterfaces api(fs::absolute(argv[7]));
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
        auto *matrixScoreTensor = resources.Describe(queries*shards, docs/shards, ACL_FLOAT, scoreMemory);
        void *partialScoreMemory = resources.Allocate(partialBytes);
        void *partialIdMemory = resources.Allocate(partialBytes);
        auto *partialScore2D = resources.Describe(queries*shards, k, ACL_FLOAT, partialScoreMemory);
        auto *partialId2D = resources.Describe(queries*shards, k, ACL_INT32, partialIdMemory);
        auto *partialScore3D = Describe3D(resources, queries, shards, k, ACL_FLOAT, partialScoreMemory);
        auto *partialId3D = Describe3D(resources, queries, shards, k, ACL_INT32, partialIdMemory);
        void *topScoreMemory = resources.Allocate(topBytes);
        void *topIdMemory = resources.Allocate(topBytes);
        auto *topScoreTensor = resources.Describe(queries, k, ACL_FLOAT, topScoreMemory);
        auto *topIdTensor = resources.Describe(queries, k, ACL_INT32, topIdMemory);
        std::vector<float> actualPartial(static_cast<size_t>(partialCount), std::numeric_limits<float>::quiet_NaN());
        std::vector<int32_t> actualLocalIds(static_cast<size_t>(partialCount), -1);
        Require(aclrtMemcpy(partialScoreMemory, partialBytes, actualPartial.data(), partialBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison initial partial scores");
        Require(aclrtMemcpy(partialIdMemory, partialBytes, actualLocalIds.data(), partialBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison initial local IDs");
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
            Require(aclrtMemcpy(actualPartial.data(), partialBytes, partialScoreMemory, partialBytes, ACL_MEMCPY_DEVICE_TO_HOST), "debug partial download");
            Require(aclrtMemcpy(actualLocalIds.data(), partialBytes, partialIdMemory, partialBytes, ACL_MEMCPY_DEVICE_TO_HOST), "debug local ID download");
            Require(aclrtMemcpy(actualContribution.data(), valueBytes, contributionMemory, valueBytes, ACL_MEMCPY_DEVICE_TO_HOST), "debug contribution download");
            Require(aclrtMemcpy(actualScores.data(), scoreBytes, scoreMemory, scoreBytes, ACL_MEMCPY_DEVICE_TO_HOST), "debug full score download");
            for (float value : actualContribution)
                if (!std::isfinite(value) || value < 0) throw std::runtime_error("Invalid device contribution");
            for (float value : actualScores)
                if (!std::isfinite(value) || value < 0) throw std::runtime_error("Invalid device score");
            ValidateOutput(actualScores, actualPartial, actualLocalIds, queries*shards, docs/shards, k);
            ValidateOutput(actualScores, actualTop, actualIds, queries, docs, k);
        };
        // The framework consumes normal executors. Obtain new ones on EVERY iteration.
        // Keep four separate workspace allocations until the final stream synchronization.
        uint64_t capacity[4] = {0, 0, 0, 0};
        void *workspace[4] = {nullptr, nullptr, nullptr, nullptr};
        auto iteration = [&](bool allocate) {
            const auto begin = std::chrono::steady_clock::now();
            if (timingMode == "with_transfers") upload();
            const auto prepareStart = std::chrono::steady_clock::now();
            uint64_t size[4] = {0, 0, 0, 0};
            aclOpExecutor *executor[4] = {nullptr, nullptr, nullptr, nullptr};
            Require(api.prepareContribution(inputTensors[0], inputTensors[1], inputTensors[2], contributionTensor,
                                            &size[0], &executor[0]), "contribution prepare");
            Require(api.prepareAccumulate(contributionTensor, offsetTensor, flatScoreTensor,
                                          &size[1], &executor[1]), "accumulate prepare");
            Require(api.prepareTopK(matrixScoreTensor, k, partialScore2D, partialId2D,
                                   &size[2], &executor[2]), "local Top-K prepare");
            Require(api.prepareMerge(partialScore3D, partialId3D, k, docs, topScoreTensor, topIdTensor,
                                     &size[3], &executor[3]), "merge prepare");
            if (expectedTopkScratch != 0 && (size[0] > UINT64_MAX - expectedTopkScratch ||
                size[2] < size[0] + expectedTopkScratch ||
                (topkImplementation == "warp" && size[2] != size[0] + expectedTopkScratch)))
                throw std::runtime_error("Local Top-K workspace mismatch; stop before execution");
            if (topkImplementation == "radix" && size[2] != expectedRadixWorkspace)
                throw std::runtime_error("Radix workspace differs from accepted independent device validation");
            for (int i = 0; i < 4; ++i) {
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
            if (allocate) Stage("execute_contribution");
            Require(api.executeContribution(workspace[0], size[0], executor[0], runtime.stream), "contribution execute"); executor[0] = nullptr;
            if (allocate) Stage("execute_accumulate");
            Require(api.executeAccumulate(workspace[1], size[1], executor[1], runtime.stream), "accumulate execute"); executor[1] = nullptr;
            if (allocate) Stage("execute_local_topk");
            Require(api.executeTopK(workspace[2], size[2], executor[2], runtime.stream), "Top-K execute"); executor[2] = nullptr;
            if (allocate) Stage("execute_merge");
            Require(api.executeMerge(workspace[3], size[3], executor[3], runtime.stream), "merge execute"); executor[3] = nullptr;
            if (allocate) Stage("synchronize_four_stage_pipeline");
            Require(aclrtSynchronizeStream(runtime.stream), "four-stage pipeline synchronize");
            const auto launchEnd = std::chrono::steady_clock::now();
            if (timingMode == "with_transfers") downloadTop();
            const auto end = std::chrono::steady_clock::now();
            // Return host wall times. launch_sync is NOT pure device-kernel time.
            return Sample{
                std::chrono::duration<double, std::milli>(end - begin).count(),
                std::chrono::duration<double, std::milli>(launchStart - prepareStart).count(),
                std::chrono::duration<double, std::milli>(launchEnd - launchStart).count()};
        };
        Stage("bootstrap_upload_prepare_execute");
        if (timingMode != "with_transfers") upload();
        const auto cold = iteration(true);
        Stage("bootstrap_debug_download");
        downloadDebug();
        const auto baselinePartial = actualPartial;
        const auto baselineLocalIds = actualLocalIds;
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
            if (!EqualBytes(actualPartial, baselinePartial) || !EqualBytes(actualLocalIds, baselineLocalIds))
                throw std::runtime_error("Repeated local candidates changed");
            if (!EqualBytes(actualContribution, baselineContribution) || !EqualBytes(actualScores, baselineScores))
                throw std::runtime_error("Repeated contribution/score bytes changed");
        };
        // 0xffffffff is NaN for IEEE FP32 and -1 for int32: overwritten outputs are required.
        std::vector<uint32_t> poison(std::max({tf.size(), static_cast<size_t>(groupCount), static_cast<size_t>(topCount), static_cast<size_t>(partialCount)}), UINT32_MAX);
        auto poisonOutputs = [&]() {
            Require(aclrtMemcpy(partialScoreMemory, partialBytes, poison.data(), partialBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison partial scores");
            Require(aclrtMemcpy(partialIdMemory, partialBytes, poison.data(), partialBytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison local IDs");
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
        Write(outputDir / "partial_scores.bin", actualPartial);
        Write(outputDir / "partial_ids.bin", actualLocalIds);
        fs::create_directory(outputDir / "bootstrap");
        Write(outputDir / "bootstrap/partial_scores.bin", baselinePartial);
        Write(outputDir / "bootstrap/partial_ids.bin", baselineLocalIds);
        Write(outputDir / "bootstrap/contributions.bin", baselineContribution);
        Write(outputDir / "bootstrap/scores.bin", baselineScores);
        Write(outputDir / "bootstrap/top_scores.bin", baselineTop);
        Write(outputDir / "bootstrap/top_ids.bin", baselineIds);
        SaveTiming(outputDir / "npu_timing.json", samples, cold, timingMode, device, queries, docs, k,
                   warmup, repeats, shards, capacity, topkImplementation, scratch);
        std::cout << "device=" << device << " mode=" << timingMode << " measured_batches=" << repeats
                  << " validated_full_repeat_batches=" << warmup + 1 << " intermediate_host_roundtrips=0\n"
                  << "median_batch_path_ms=" << Percentile(samples, .5, 0)
                  << "\nmedian_prepare_ms=" << Percentile(samples, .5, 1)
                  << "\nmedian_four_launches_and_sync_ms=" << Percentile(samples, .5, 2) << '\n';

        std::cout << "Repeated call completed; check independent CPU formula and all-query cross-device ranking. "
                  << "Prepared-input experiment, not complete retrieval or a speedup claim.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
