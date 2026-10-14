// Own grouped BM25 score reduction. Not yet compiled or executed on A5.
#include <algorithm>
#include <cstdint>
#include <limits>
#include "bm25_accumulate_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kThreads = 256;

ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    const auto *values = context->GetInputShape(0);
    const auto *offsets = context->GetInputShape(1);
    auto *scores = context->GetOutputShape(0);
    if (values == nullptr || offsets == nullptr || scores == nullptr ||
        values->GetDimNum() != 1 || offsets->GetDimNum() != 1 ||
        values->GetDim(0) <= 0 || offsets->GetDim(0) < 2 ||
        offsets->GetDim(0) - 1 > std::numeric_limits<int32_t>::max()) return ge::GRAPH_FAILED;
    *scores = gert::Shape({offsets->GetDim(0) - 1});
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
    if (context == nullptr || context->GetRawTilingData() == nullptr) return ge::GRAPH_FAILED;
    const auto *values = context->GetInputShape(0);
    const auto *offsets = context->GetInputShape(1);
    if (values == nullptr || offsets == nullptr || values->GetStorageShape().GetDimNum() != 1 ||
        offsets->GetStorageShape().GetDimNum() != 1) return ge::GRAPH_FAILED;
    const int64_t valueCount = values->GetStorageShape().GetDim(0);
    const int64_t groupCount = offsets->GetStorageShape().GetDim(0) - 1;
    if (valueCount <= 0 || groupCount <= 0 || groupCount > std::numeric_limits<int32_t>::max())
        return ge::GRAPH_FAILED;
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0) return ge::GRAPH_FAILED;
    const uint32_t blocks = std::min(cores, (static_cast<uint32_t>(groupCount) - 1) / kThreads + 1);
    optiling::Bm25AccumulateTilingData data;
    data.set_groupCount(static_cast<uint32_t>(groupCount));
    data.set_valueCount(static_cast<uint64_t>(valueCount));
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
class Bm25Accumulate : public OpDef {
public:
    explicit Bm25Accumulate(const char *name) : OpDef(name)
    {
        this->Input("values").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("offsets").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(Tiling);
        this->AICore().AddConfig("ascend950");
    }
};
OP_ADD(Bm25Accumulate);
}
