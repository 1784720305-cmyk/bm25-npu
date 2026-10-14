// Own deterministic Top-K. Correctness-first version; pending A5 compilation/execution.
#include <algorithm>
#include <cstdint>
#include <limits>
#include "bm25_top_k_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kThreads = 256;
constexpr int64_t kMaxK = 32;

bool ValidDimensions(int64_t queries, int64_t docs, int64_t k)
{
    return queries > 0 && queries <= std::numeric_limits<int32_t>::max() &&
           docs > 0 && docs <= std::numeric_limits<int32_t>::max() && k >= 1 && k <= kMaxK && k <= docs;
}

ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetAttrs() == nullptr) return ge::GRAPH_FAILED;
    const auto *input = context->GetInputShape(0);
    const int64_t *k = context->GetAttrs()->GetInt(0);
    auto *topScores = context->GetOutputShape(0);
    auto *topIds = context->GetOutputShape(1);
    if (input == nullptr || input->GetDimNum() != 2 || k == nullptr ||
        topScores == nullptr || topIds == nullptr ||
        !ValidDimensions(input->GetDim(0), input->GetDim(1), *k)) return ge::GRAPH_FAILED;
    *topScores = gert::Shape({input->GetDim(0), *k});
    *topIds = *topScores;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling(gert::TilingContext *context)
{
    if (context == nullptr || context->GetRawTilingData() == nullptr ||
        context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr) return ge::GRAPH_FAILED;
    const auto *input = context->GetInputShape(0);
    const auto *outScores = context->GetOutputShape(0);
    const auto *outIds = context->GetOutputShape(1);
    const int64_t *k = context->GetAttrs()->GetInt(0);
    if (input == nullptr || outScores == nullptr || outIds == nullptr || k == nullptr ||
        input->GetStorageShape().GetDimNum() != 2) return ge::GRAPH_FAILED;
    const auto &shape = input->GetStorageShape();
    const int64_t queries = shape.GetDim(0);
    const int64_t docs = shape.GetDim(1);
    if (!ValidDimensions(queries, docs, *k)) return ge::GRAPH_FAILED;
    // Reject wrong output sizes before the kernel can write outside the supplied tensors.
    for (const auto *output : {outScores, outIds}) {
        const auto &outShape = output->GetStorageShape();
        if (outShape.GetDimNum() != 2 || outShape.GetDim(0) != queries || outShape.GetDim(1) != *k)
            return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0) return ge::GRAPH_FAILED;
    const auto queryCount = static_cast<uint32_t>(queries);
    const uint32_t blocks = std::min(cores, (queryCount - 1) / kThreads + 1);
    optiling::Bm25TopKTilingData data;
    data.set_queryCount(queryCount);
    data.set_docCount(static_cast<uint32_t>(docs));
    data.set_k(static_cast<uint32_t>(*k));
    data.set_blockNum(blocks);
    context->SetBlockDim(blocks);
    auto *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) return ge::GRAPH_FAILED;
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Bm25TopK : public OpDef {
public:
    explicit Bm25TopK(const char *name) : OpDef(name)
    {
        this->Input("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("top_scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("top_ids").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("k").AttrType(REQUIRED).Int();
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(Tiling);
        this->AICore().AddConfig("ascend950");
    }
};
OP_ADD(Bm25TopK);
}
