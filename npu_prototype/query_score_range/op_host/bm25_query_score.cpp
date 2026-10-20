// Independent contiguous-range short-query experiment. Formula/IR match the accepted score operator.
// Source only: requires the target CANN compiler and Ascend950 validation.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "bm25_query_score_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kDocTile = 2048;
constexpr uint32_t kScratchBytes = 9568; // score tile, 64 cursors/ends/terms, warp totals/control
constexpr uint32_t kSimtLocalBytes = 32 * 1024;
constexpr int64_t kLimit = std::numeric_limits<int32_t>::max();
struct Dimensions { uint32_t queries, docs, vocab, postings, queryStorage; };

bool DecodeShapes(const gert::Shape *const (&shapes)[7], Dimensions &dims)
{
    int64_t length[7];
    for (std::size_t i = 0; i < 7; ++i) {
        if (shapes[i] == nullptr || shapes[i]->GetDimNum() != 1) return false;
        length[i] = shapes[i]->GetDim(0);
    }
    if (length[5] < 2 || length[5] > kLimit + 1) return false;
    const int64_t docs = length[3], vocab = length[4], queries = length[5] - 1;
    if (docs < 1 || docs > kLimit || vocab < 1 || vocab >= kLimit ||
        queries < 1 || queries > kLimit / docs ||
        length[0] != vocab + 1 || length[1] < 1 || length[1] > kLimit ||
        length[2] != length[1] || length[6] < 1 || length[6] > kLimit) return false;
    // Empty logical word lists use one unused physical INT32 element; offsets stay all zero.
    dims = {static_cast<uint32_t>(queries), static_cast<uint32_t>(docs),
            static_cast<uint32_t>(vocab), static_cast<uint32_t>(length[1]),
            static_cast<uint32_t>(length[6])};
    return true;
}

ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    const gert::Shape *shapes[7];
    for (std::size_t i = 0; i < 7; ++i) shapes[i] = context->GetInputShape(i);
    Dimensions dims{};
    auto *output = context->GetOutputShape(0);
    if (output == nullptr || !DecodeShapes(shapes, dims)) return ge::GRAPH_FAILED;
    *output = gert::Shape({dims.queries, dims.docs});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling(gert::TilingContext *context)
{
    if (context == nullptr || context->GetRawTilingData() == nullptr ||
        context->GetPlatformInfo() == nullptr) return ge::GRAPH_FAILED;
    const gert::Shape *shapes[7];
    for (std::size_t i = 0; i < 7; ++i) {
        const auto *input = context->GetInputShape(i);
        if (input == nullptr) return ge::GRAPH_FAILED;
        shapes[i] = &input->GetStorageShape();
    }
    Dimensions dims{};
    const auto *output = context->GetOutputShape(0);
    if (!DecodeShapes(shapes, dims) || output == nullptr) return ge::GRAPH_FAILED;
    const auto &shape = output->GetStorageShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(0) != dims.queries || shape.GetDim(1) != dims.docs)
        return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (cores == 0 || ubBytes < kScratchBytes + kSimtLocalBytes) return ge::GRAPH_FAILED;
    const uint32_t tiles = (dims.docs - 1) / kDocTile + 1;
    const uint32_t partitions = std::min(tiles, std::max(1U, cores / std::min(cores, dims.queries)));
    const uint64_t tasks = static_cast<uint64_t>(dims.queries) * partitions;
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(cores, tasks));
    optiling::Bm25QueryScoreTilingData data;
    data.set_queryCount(dims.queries);
    data.set_docCount(dims.docs);
    data.set_vocabSize(dims.vocab);
    data.set_postingCount(dims.postings);
    data.set_queryStorageCount(dims.queryStorage);
    data.set_partitionsPerQuery(partitions);
    data.set_blockNum(blocks);
    const uint64_t systemBytes = platform.GetLibApiWorkSpaceSize();
    if (systemBytes > std::numeric_limits<std::size_t>::max()) return ge::GRAPH_FAILED;
    auto *workspace = context->GetWorkspaceSizes(1);
    auto *raw = context->GetRawTilingData();
    if (workspace == nullptr || raw->GetData() == nullptr || data.GetDataSize() > raw->GetCapacity())
        return ge::GRAPH_FAILED;
    workspace[0] = static_cast<std::size_t>(systemBytes); // No per-query-expanded GM workspace.
    context->SetBlockDim(blocks);
    context->SetLocalMemorySize(kSimtLocalBytes);
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Bm25QueryScore : public OpDef {
public:
    explicit Bm25QueryScore(const char *name) : OpDef(name)
    {
        this->Input("term_offsets").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("posting_doc_ids").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("posting_tf").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("doc_norm").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("idf").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("query_offsets").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("query_term_ids").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(Tiling);
        this->AICore().AddConfig("ascend950");
    }
};
OP_ADD(Bm25QueryScore);
}
