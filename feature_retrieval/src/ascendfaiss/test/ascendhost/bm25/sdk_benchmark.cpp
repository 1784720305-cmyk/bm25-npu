#include "ascendhost/include/index/AscendIndexBm25.h"
#include "ascendhost/src/bm25/runtime.h"
#include "common/bm25/search_inputs.h"
#include "common/bm25/validation.h"
#include "test/ascendhost/bm25/input_io.h"
#include "ascenddaemon/AscendResourcesProxy.h"
#include "common/utils/SocUtils.h"
#include "common/ErrorCode.h"
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ratio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using ascend::bm25::Check;
using faiss::ascend::AscendIndexBm25;
using faiss::ascend::Bm25SearchResult;
using Clock = std::chrono::steady_clock;
struct Sample {
    int cycle;
    int which;
    double ms;
};

void Expect(AscendIndexBm25 &model, ascend::APP_ERROR code)
{
    if (code != ascend::APP_ERR_OK) {
        const auto error = model.GetLastError();
        throw std::runtime_error("SDK BM25 status=" + std::to_string(code) + " acl=" + std::to_string(error.aclCode) +
                                 " " + error.message.data());
    }
}

void Compare(const Bm25SearchResult &selected, const std::vector<int32_t> &ids, const std::vector<float> &scores)
{
    Check(ids.size() == scores.size() && selected.labels.size() == ids.size() &&
              selected.scores.size() == scores.size(),
          "SDK TopK shape differs");
    for (size_t i = 0; i < ids.size(); ++i) {
        Check(selected.labels[i] == ids[i], "SDK strict oracle ID differs");
        Check(std::isfinite(selected.scores[i]) && selected.scores[i] >= 0.0F &&
                  std::abs(static_cast<double>(selected.scores[i]) - scores[i]) <= 1e-6 + 2e-6 * std::abs(scores[i]),
              "SDK oracle score differs");
    }
}

void Same(const Bm25SearchResult &current, const Bm25SearchResult &first)
{
    Check(current.labels == first.labels && current.scores.size() == first.scores.size() &&
              std::memcmp(current.scores.data(), first.scores.data(), first.scores.size() * sizeof(float)) == 0,
          "SDK repeated result bits differ");
}

uint64_t Words(const bm25search::Queries &queries)
{
    uint64_t words = 0;
    for (const auto &query : queries) {
        words += query.size();
    }
    return words;
}
} // namespace

int main(int argc, char **argv)
{
    try {
        Check(argc == 7, "Usage: sdk_benchmark INPUT VENDORS DEVICE NEW_OUTPUT WARMUP REPEATS");
        const auto input = std::filesystem::absolute(argv[1]);
        const auto output = std::filesystem::absolute(argv[4]);
        Check(!std::filesystem::exists(output), "New SDK benchmark output required");
        const int device = static_cast<int>(ascend::bm25::Number(argv[3], 0, INT32_MAX));
        const int warmup = static_cast<int>(ascend::bm25::Number(argv[5], 1, 10000));
        const int repeats = static_cast<int>(ascend::bm25::Number(argv[6], 10, 100000));
        const auto index = ascend::bm25::LoadIndex(input / "index");
        const auto a = bm25search::LoadQueries((input / "A.txt").string());
        const auto b = bm25search::LoadQueries((input / "B.txt").string());
        Check(!a.empty() && a.size() == b.size(), "SDK A/B packet shape differs");
        const auto idsA = ascend::bm25::Read<int32_t>(input / "A/expected_top_ids.bin");
        const auto idsB = ascend::bm25::Read<int32_t>(input / "B/expected_top_ids.bin");
        const auto scoresA = ascend::bm25::Read<float>(input / "A/expected_top_scores.bin");
        const auto scoresB = ascend::bm25::Read<float>(input / "B/expected_top_scores.bin");
        Check(!idsA.empty() && idsA.size() % a.size() == 0 && idsA.size() == idsB.size(), "SDK oracle shape differs");
        Check(faiss::ascend::SocUtils::GetInstance().GetDeviceCount() > static_cast<uint32_t>(device),
              "SDK device is unavailable");
        ascend::bm25::RequireAcl(aclrtSetDevice(device), "select SDK benchmark device");
        const auto resources = std::make_shared<ascend::AscendResourcesProxy>();
        resources->initialize();
        const auto streamOwner = resources->getDefaultStream();
        Check(streamOwner != nullptr && streamOwner->GetStream() != nullptr, "SDK default stream unavailable");
        faiss::ascend::Bm25Config config;
        config.deviceId = device;
        config.resources = resources;
        config.vendorDirectory = std::filesystem::absolute(argv[2]).string();
        config.queryCount = a.size();
        config.maxQueryWords = std::max(Words(a), Words(b));
        config.topK = idsA.size() / a.size();
        ascend::bm25::RequireAcl(aclrtGetCurrentContext(&config.context), "get SDK benchmark context");
        Check(config.context != nullptr, "SDK context unavailable");
        AscendIndexBm25 model;
        Expect(model, model.Init(index, config));
        Bm25SearchResult firstA;
        Bm25SearchResult firstB;
        Expect(model, model.Search(a, firstA));
        Compare(firstA, idsA, scoresA);
        Expect(model, model.Search(b, firstB));
        Compare(firstB, idsB, scoresB);
        for (int cycle = 0; cycle < warmup; ++cycle) {
            for (int slot = 0; slot < 2; ++slot) {
                const int which = slot ^ (cycle & 1);
                Bm25SearchResult result;
                Expect(model, model.Search(which == 0 ? a : b, result));
                Same(result, which == 0 ? firstA : firstB);
            }
        }
        std::vector<Sample> samples;
        samples.reserve(static_cast<size_t>(repeats) * 2);
        for (int cycle = 0; cycle < repeats; ++cycle) {
            for (int slot = 0; slot < 2; ++slot) {
                const int which = slot ^ (cycle & 1);
                Bm25SearchResult result;
                const auto start = Clock::now();
                const auto code = model.Search(which == 0 ? a : b, result);
                const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                Expect(model, code);
                Check(elapsed > 0.0 && std::isfinite(elapsed), "Invalid SDK timing sample");
                Same(result, which == 0 ? firstA : firstB);
                samples.push_back({cycle, which, elapsed});
            }
        }
        Expect(model, model.Close());
        aclrtContext after = nullptr;
        int32_t afterDevice = -1;
        ascend::bm25::RequireAcl(aclrtGetCurrentContext(&after), "SDK context after benchmark Close");
        ascend::bm25::RequireAcl(aclrtGetDevice(&afterDevice), "SDK device after benchmark Close");
        Check(after == config.context && afterDevice == device, "BM25 changed SDK context/device ownership");
        ascend::bm25::RequireAcl(aclrtSynchronizeStream(streamOwner->GetStream()), "SDK stream after benchmark Close");
        Check(std::filesystem::create_directories(output), "Cannot create SDK output");
        ascend::bm25::Write(output / "A_scores.bin", firstA.scores);
        ascend::bm25::Write(output / "A_ids.bin", firstA.labels);
        ascend::bm25::Write(output / "B_scores.bin", firstB.scores);
        ascend::bm25::Write(output / "B_ids.bin", firstB.labels);
        std::ofstream report(output / "report.json");
        report << std::setprecision(17)
               << "{\"status\":\"actual_integrated_SDK_public_Search_and_shared_context_checked\",\"device\":" << device
               << ",\"queries\":" << a.size() << ",\"k\":" << config.topK << ",\"warmup\":" << warmup
               << ",\"repeats\":" << repeats
               << ",\"timer\":\"caller steady_clock around AscendIndexBm25::Search including returned INT64 labels\","
                  "\"SDK_context_device_stream_survived_Close\":true,\"samples\":[";
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto &sample = samples[i];
            report << (i ? "," : "") << "{\"cycle\":" << sample.cycle << ",\"case\":\""
                   << (sample.which == 0 ? 'A' : 'B') << "\",\"call_wall_ms\":" << sample.ms << "}";
        }
        report << "]}\n";
        report.close();
        Check(static_cast<bool>(report), "Cannot write SDK report");
        std::cout << "PASS actual integrated SDK public Search, strict oracle, repeated bits and SDK lifetime\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
