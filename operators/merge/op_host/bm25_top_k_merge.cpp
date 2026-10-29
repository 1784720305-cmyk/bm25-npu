// Self-developed merge of ordered document-shard candidates. Pending A5 validation.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "bm25_top_k_merge_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kMaxShards = 65536;
constexpr int64_t kMaxK = 32;

bool ValidShape(const gert::Shape &scores,const gert::Shape &ids,int64_t k,int64_t docs)
{
    if(scores.GetDimNum()!=3 || ids.GetDimNum()!=3) return false;
    for(std::size_t d=0;d<3;++d) if(scores.GetDim(d)!=ids.GetDim(d))return false;
    const int64_t q=scores.GetDim(0),s=scores.GetDim(1),limit=std::numeric_limits<int32_t>::max();
    return docs>=k && docs<=limit && q>=1 && q<=limit/docs && s>=1 && s<=kMaxShards &&
        k>=1 && k<=kMaxK && scores.GetDim(2)==k;
}

ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetAttrs() == nullptr) return ge::GRAPH_FAILED;
    const auto *scores = context->GetInputShape(0);
    const auto *ids = context->GetInputShape(1);
    const int64_t *k = context->GetAttrs()->GetInt(0);
    const int64_t *docs = context->GetAttrs()->GetInt(1);
    auto *outScores = context->GetOutputShape(0);
    auto *outIds = context->GetOutputShape(1);
    if (scores == nullptr || ids == nullptr || k == nullptr || docs == nullptr ||
        outScores == nullptr || outIds == nullptr || !ValidShape(*scores, *ids, *k, *docs))
        return ge::GRAPH_FAILED;
    *outScores = gert::Shape({scores->GetDim(0), (scores->GetDim(1)+31)/32, *k});
    *outIds = *outScores;
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
    if (context == nullptr || context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr ||
        context->GetRawTilingData() == nullptr) return ge::GRAPH_FAILED;
    const auto *scores = context->GetInputShape(0);
    const auto *ids = context->GetInputShape(1);
    const auto *outScores = context->GetOutputShape(0);
    const auto *outIds = context->GetOutputShape(1);
    const int64_t *k = context->GetAttrs()->GetInt(0);
    const int64_t *docs = context->GetAttrs()->GetInt(1);
    if (scores == nullptr || ids == nullptr || outScores == nullptr || outIds == nullptr ||
        k == nullptr || docs == nullptr || !ValidShape(scores->GetStorageShape(), ids->GetStorageShape(), *k, *docs))
        return ge::GRAPH_FAILED;
    const auto &shape = scores->GetStorageShape();
    for (const auto *out : {outScores, outIds}) {
        const auto &outShape = out->GetStorageShape();
        if (outShape.GetDimNum() != 3 || outShape.GetDim(0) != shape.GetDim(0) ||
            outShape.GetDim(1) != (shape.GetDim(1)+31)/32 || outShape.GetDim(2) != *k)
            return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0) return ge::GRAPH_FAILED;
    const uint32_t queries = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t blocks = std::min<uint64_t>(cores, static_cast<uint64_t>(queries)*((shape.GetDim(1)+31)/32));
    optiling::Bm25TopKMergeTilingData data;
    data.set_queryCount(queries);
    data.set_shardCount(static_cast<uint32_t>(shape.GetDim(1)));
    data.set_docsPerShard(static_cast<uint32_t>(*docs)); // global ID range; no per-list offset
    data.set_k(static_cast<uint32_t>(*k));
    data.set_blockNum(blocks);
    context->SetBlockDim(blocks);
    auto *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) return ge::GRAPH_FAILED;
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    if (data.GetDataSize() > context->GetRawTilingData()->GetCapacity()) return ge::GRAPH_FAILED;
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Bm25TopKMerge : public OpDef {
public:
    explicit Bm25TopKMerge(const char *name) : OpDef(name)
    {
        this->Input("partial_scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("partial_ids").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("top_scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("top_ids").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("k").AttrType(REQUIRED).Int();
        this->Attr("doc_count").AttrType(REQUIRED).Int();
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(Tiling);
        this->AICore().AddConfig("ascend950");
    }
};
OP_ADD(Bm25TopKMerge);
}
