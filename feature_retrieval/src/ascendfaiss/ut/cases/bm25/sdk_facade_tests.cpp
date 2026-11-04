#include "ascendhost/include/index/AscendIndexBm25.h"
#include "ascendhost/src/bm25/large_search.h"
#include "common/ErrorCode.h"
#include "common/bm25/index_data.h"
#include "common/bm25/validation.h"
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

namespace ascend {
class AscendResourcesProxy {};
} // namespace ascend

int main()
{
    using ascend::APP_ERR_OK;
    using ascend::APP_ERR_INVALID_PARAM;
    using ascend::APP_ERR_ILLEGAL_OPERATION;
    using ascend::APP_ERR_INNER_ERROR;
    using ascend::AscendResourcesProxy;
    using ascend::bm25::Check;
    using ascend::bm25::Index;
    using faiss::ascend::AscendIndexBm25;
    using faiss::ascend::Bm25Config;
    using faiss::ascend::Bm25SearchResult;
    try {
        AscendIndexBm25 model;
        Bm25SearchResult output{{9}, {9}};
        Check(model.Search({{0}}, output) == APP_ERR_ILLEGAL_OPERATION, "Search before Init accepted");
        Index input({0, 1}, {0}, {1}, {1}, {2});
        Bm25Config config;
        Check(model.Init(input, config) == APP_ERR_INVALID_PARAM, "Missing SDK binding accepted");
        config.context = reinterpret_cast<void *>(1);
        config.resources = std::make_shared<AscendResourcesProxy>();
        config.queryCount = config.topK = config.maxQueryWords = 1;
        config.vendorDirectory = "fail_init";
        Check(model.Init(input, config) == APP_ERR_INNER_ERROR && model.GetLastError().aclCode == 42,
              "Init failure did not preserve ACL error");
        config.vendorDirectory = "vendors";
        Check(model.Init(input, config) == APP_ERR_OK, "Retry Init failed");
        Check(model.Init(input, config) == APP_ERR_ILLEGAL_OPERATION, "Double Init replaced active state");
        input.idf[0] = 99;
        std::weak_ptr<AscendResourcesProxy> owner = config.resources;
        config.resources.reset();
        Check(!owner.expired(), "Facade lost SDK resource lifetime");
        Check(model.Search({{-1}}, output) == APP_ERR_INVALID_PARAM && output.scores[0] == 9 && output.labels[0] == 9,
              "Rejected query modified output");
        Check(model.Search({{0}}, output) == APP_ERR_OK && output.scores == std::vector<float>{2} &&
                  output.labels == std::vector<int64_t>{0},
              "Snapshot/recovery/output conversion failed");
        bm25test::failSearch = true;
        Check(model.Search({{0}}, output) == APP_ERR_INNER_ERROR && model.GetLastError().aclCode == 161001 &&
                  output.scores[0] == 2,
              "Device failure changed output or lost diagnostic");
        Check(model.Search({{0}}, output) == APP_ERR_ILLEGAL_OPERATION, "Failed session accepted another Search");
        bm25test::failClose = true;
        Check(model.Close() == APP_ERR_INNER_ERROR && !owner.expired(), "Failed Close discarded resource owner");
        bm25test::failClose = false;
        Check(model.Close() == APP_ERR_OK && owner.expired(), "Successful Close did not release state");
        Check(model.Close() == APP_ERR_OK, "Close is not idempotent");
        std::cout << "PASS: SDK facade state, immutable snapshot, error/output atomicity and Close retry\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
