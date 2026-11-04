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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using ascend::bm25::Check;
using faiss::ascend::AscendIndexBm25;
using faiss::ascend::Bm25SearchResult;

void ExpectStatus(AscendIndexBm25 &model, ascend::APP_ERROR status)
{
    if (status != ascend::APP_ERR_OK) {
        const auto error = model.GetLastError();
        throw std::runtime_error("BM25 status=" + std::to_string(status) + " acl=" + std::to_string(error.aclCode) +
                                 " " + error.message.data());
    }
}

void Compare(const Bm25SearchResult &result, const std::filesystem::path &gold)
{
    const auto ids = ascend::bm25::Read<int32_t>(gold / "expected_top_ids.bin");
    const auto scores = ascend::bm25::Read<float>(gold / "expected_top_scores.bin");
    Check(result.labels.size() == ids.size() && result.scores.size() == scores.size(), "SDK output shape differs");
    for (size_t i = 0; i < ids.size(); ++i) {
        Check(result.labels[i] == ids[i], "SDK strict ranking differs from independent oracle");
        Check(std::isfinite(result.scores[i]) && result.scores[i] >= 0 &&
                  std::abs(static_cast<double>(result.scores[i]) - scores[i]) <= 1e-6 + 2e-6 * std::abs(scores[i]),
              "SDK score differs from independent oracle");
    }
}

uint64_t WordCount(const bm25search::Queries &queries)
{
    uint64_t count = 0;
    for (const auto &row : queries) {
        count += row.size();
    }
    return count;
}
} // namespace

int main(int argc, char **argv)
{
    try {
        Check(argc == 5, "Usage: sdk_shared_check INPUT VENDORS DEVICE NEW_REPORT_JSON");
        const auto input = std::filesystem::absolute(argv[1]);
        const auto output = std::filesystem::absolute(argv[4]);
        Check(!std::filesystem::exists(output), "New shared-check report required");
        const int device = static_cast<int>(ascend::bm25::Number(argv[3], 0, INT32_MAX));
        Check(faiss::ascend::SocUtils::GetInstance().GetDeviceCount() > static_cast<uint32_t>(device),
              "SDK did not discover the requested device");
        ascend::bm25::RequireAcl(aclrtSetDevice(device), "select SDK device");
        const auto resources = std::make_shared<ascend::AscendResourcesProxy>();
        resources->initialize();
        const auto sdkStreamOwner = resources->getDefaultStream();
        Check(sdkStreamOwner != nullptr, "SDK stream resource unavailable");
        const auto sdkStream = sdkStreamOwner->GetStream();
        Check(sdkStream != nullptr, "SDK default stream unavailable");
        ascend::bm25::RequireAcl(aclrtSynchronizeStream(sdkStream), "SDK stream before BM25 Init");
        const auto index = ascend::bm25::LoadIndex(input / "index");
        const auto a = bm25search::LoadQueries((input / "A.txt").string());
        const auto b = bm25search::LoadQueries((input / "B.txt").string());
        Check(a.size() == b.size() && !a.empty(), "A/B query shapes differ");
        const auto goldIds = ascend::bm25::Read<int32_t>(input / "A/expected_top_ids.bin");
        Check(goldIds.size() % a.size() == 0, "Oracle query shape differs");
        faiss::ascend::Bm25Config config;
        config.deviceId = device;
        config.resources = resources;
        config.vendorDirectory = std::filesystem::absolute(argv[2]).string();
        config.queryCount = a.size();
        config.maxQueryWords = std::max(WordCount(a), WordCount(b));
        config.topK = goldIds.size() / a.size();
        ascend::bm25::RequireAcl(aclrtGetCurrentContext(&config.context), "get SDK context");
        Check(config.context != nullptr, "SDK current context unavailable");
        const auto references = resources.use_count();
        AscendIndexBm25 model;
        ExpectStatus(model, model.Init(index, config));
        Check(resources.use_count() > references, "BM25 did not retain the SDK resource proxy");
        Bm25SearchResult first;
        ExpectStatus(model, model.Search(a, first));
        Compare(first, input / "A");
        auto rejected = a;
        rejected.front() = {-1};
        auto unchanged = first;
        Check(model.Search(rejected, unchanged) == ascend::APP_ERR_INVALID_PARAM && unchanged.labels == first.labels &&
                  unchanged.scores == first.scores,
              "Invalid SDK query changed output or returned incorrect status");
        Bm25SearchResult next;
        ExpectStatus(model, model.Search(b, next));
        Compare(next, input / "B");
        ExpectStatus(model, model.Search(a, next));
        Compare(next, input / "A");
        Check(next.labels == first.labels && next.scores.size() == first.scores.size() &&
                  std::memcmp(next.scores.data(), first.scores.data(), first.scores.size() * sizeof(float)) == 0,
              "SDK repeated query bits changed");
        ExpectStatus(model, model.Close());
        Check(resources.use_count() == references, "Closed BM25 retained an unnecessary SDK proxy reference");
        // Exercise another lifecycle using the same SDK resources/context.
        ExpectStatus(model, model.Init(index, config));
        ExpectStatus(model, model.Search(a, next));
        Compare(next, input / "A");
        ExpectStatus(model, model.Close());
        aclrtContext after = nullptr;
        int32_t afterDevice = -1;
        ascend::bm25::RequireAcl(aclrtGetCurrentContext(&after), "read context after BM25 Close");
        ascend::bm25::RequireAcl(aclrtGetDevice(&afterDevice), "read device after BM25 Close");
        Check(after == config.context && afterDevice == device, "BM25 changed SDK context/device ownership");
        ascend::bm25::RequireAcl(aclrtSynchronizeStream(sdkStream), "SDK stream after BM25 Close");
        std::ofstream report(output);
        report << "{\"status\":\"shared_SDK_context_oracle_retry_lifetime_and_stream_passed\",\"device\":" << device
               << ",\"queries\":" << a.size() << ",\"k\":" << config.topK << ",\"init_close_cycles\":2}\n";
        report.close();
        Check(static_cast<bool>(report), "Cannot write shared SDK acceptance report");
        std::cout << "PASS: actual SDK resource proxy, A/B oracle, rejected-query recovery and surviving SDK stream\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
