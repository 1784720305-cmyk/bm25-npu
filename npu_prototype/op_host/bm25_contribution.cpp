// A5 prototype: not compiled or executed on the target yet.
#include <algorithm>
#include <cstdint>
#include <limits>
#include "bm25_contribution_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kThreads = 256;

ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    const auto *tf = context->GetInputShape(0);
    const auto *norm = context->GetInputShape(1);
    const auto *idf = context->GetInputShape(2);
    auto *output = context->GetOutputShape(0);
    if (tf == nullptr || norm == nullptr || idf == nullptr || output == nullptr ||
        tf->GetDimNum() != 1 || norm->GetDimNum() != 1 || idf->GetDimNum() != 1 ||
        tf->GetDim(0) <= 0 || norm->GetDim(0) != tf->GetDim(0) || idf->GetDim(0) != tf->GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    *output = *tf;
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
    const auto *tf = context->GetInputShape(0);
    if (tf == nullptr || tf->GetStorageShape().GetDimNum() != 1) return ge::GRAPH_FAILED;
    const int64_t count = tf->GetStorageShape().GetDim(0);
    if (count <= 0 || count > std::numeric_limits<int32_t>::max()) return ge::GRAPH_FAILED;
    for (int i = 1; i < 3; ++i) {
        const auto *input = context->GetInputShape(i);
        if (input == nullptr || input->GetStorageShape().GetDimNum() != 1 ||
            input->GetStorageShape().GetDim(0) != count) return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0) return ge::GRAPH_FAILED;
    const uint32_t blocks = std::min(cores, (static_cast<uint32_t>(count) - 1) / kThreads + 1);
    optiling::Bm25ContributionTilingData data;
    data.set_recordCount(static_cast<uint32_t>(count));
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
class Bm25Contribution : public OpDef {
public:
    explicit Bm25Contribution(const char *name) : OpDef(name)
    {
        this->Input("tf").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("doc_norm").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("idf").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("contribution").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(Tiling);
        this->AICore().AddConfig("ascend950");
    }
};
OP_ADD(Bm25Contribution);
}
