// Isolated fused scoring/local-selection experiment. Source only; A5 compilation/validation pending.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "bm25_fused_score_top_k_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kBufferBytes = 50848;
// SetLocalMemorySize declares DYNAMIC UB accessible to the kernel, not SIMT DCache.
// The old 32 KiB declaration was smaller than our 50,848-byte TBuf. In particular,
// control at byte34,048 was already out of bounds even for a three-document shard.
constexpr uint32_t kDynamicUbBytes = 64*1024;
constexpr uint32_t kMinDCacheBytes = 32*1024;
constexpr uint32_t kCompilerReservedBytes = 8*1024;
static_assert(kDynamicUbBytes >= kBufferBytes && kDynamicUbBytes%32 == 0,
              "Declared dynamic UB must cover the complete TBuf layout");
constexpr int64_t kLimit = std::numeric_limits<int32_t>::max();
struct Dimensions { uint32_t queries, docs, vocab, postings, queryStorage, k, shards, debug; };

bool Decode(const gert::Shape *const (&shapes)[7], const int64_t *k,
            const int64_t *shards, const int64_t *debug, Dimensions &dims)
{
    if (k == nullptr || shards == nullptr || debug == nullptr || *k < 1 || *k > 32 ||
        *shards < 1 || *shards > 65536 || (*debug != 0 && *debug != 1 && *debug != 2)) return false;
    int64_t length[7];
    for (std::size_t i = 0; i < 7; ++i) {
        if (shapes[i] == nullptr || shapes[i]->GetDimNum() != 1) return false;
        length[i] = shapes[i]->GetDim(0);
    }
    if (length[5] < 2 || length[5] > kLimit+1) return false;
    const int64_t docs = length[3], vocab = length[4], queries = length[5]-1;
    if (docs < 1 || docs > kLimit || vocab < 1 || vocab >= kLimit ||
        queries < 1 || queries > kLimit/docs || length[0] != vocab*(*shards+1) ||
        length[1] < 1 || length[1] > kLimit || length[2] != length[1] ||
        length[6] < 1 || length[6] > kLimit || docs % (*shards) != 0 ||
        docs / (*shards) < *k || docs / (*shards) > 8192) return false;
    // debug2 is a selector-only diagnostic: never used by public Search.
    if (*debug == 2 && static_cast<uint64_t>(length[1]) != static_cast<uint64_t>(queries)*docs) return false;
    dims = {static_cast<uint32_t>(queries), static_cast<uint32_t>(docs),
        static_cast<uint32_t>(vocab), static_cast<uint32_t>(length[1]), static_cast<uint32_t>(length[6]),
        static_cast<uint32_t>(*k), static_cast<uint32_t>(*shards), static_cast<uint32_t>(*debug)};
    return true;
}

bool ValidOutput(const gert::Shape &shape, const Dimensions &dims, uint32_t slot)
{
    if (slot < 2) return shape.GetDimNum() == 3 && shape.GetDim(0) == dims.queries &&
        shape.GetDim(1) == dims.shards && shape.GetDim(2) == dims.k;
    if (dims.debug != 0) return shape.GetDimNum() == 2 && shape.GetDim(0) == dims.queries && shape.GetDim(1) == dims.docs;
    return shape.GetDimNum() == 1 && shape.GetDim(0) == 1;
}

ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetAttrs() == nullptr) return ge::GRAPH_FAILED;
    const gert::Shape *shapes[7];
    for (std::size_t i = 0; i < 7; ++i) shapes[i] = context->GetInputShape(i);
    Dimensions dims{};
    if (!Decode(shapes, context->GetAttrs()->GetInt(0), context->GetAttrs()->GetInt(1),
                context->GetAttrs()->GetInt(2), dims)) return ge::GRAPH_FAILED;
    for (uint32_t slot = 0; slot < 3; ++slot) {
        auto *output = context->GetOutputShape(slot);
        if (output == nullptr) return ge::GRAPH_FAILED;
        if (slot < 2) *output = gert::Shape({dims.queries, dims.shards, dims.k});
        else if (dims.debug != 0) *output = gert::Shape({dims.queries, dims.docs});
        else *output = gert::Shape({1});
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling(gert::TilingContext *context)
{
    if (context == nullptr || context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr ||
        context->GetRawTilingData() == nullptr) return ge::GRAPH_FAILED;
    const gert::Shape *shapes[7];
    for (std::size_t i = 0; i < 7; ++i) {
        const auto *input = context->GetInputShape(i);
        if (input == nullptr) return ge::GRAPH_FAILED;
        shapes[i] = &input->GetStorageShape();
    }
    Dimensions dims{};
    if (!Decode(shapes, context->GetAttrs()->GetInt(0), context->GetAttrs()->GetInt(1),
                context->GetAttrs()->GetInt(2), dims)) return ge::GRAPH_FAILED;
    for (uint32_t slot = 0; slot < 3; ++slot) {
        const auto *out = context->GetOutputShape(slot);
        if (out == nullptr || !ValidOutput(out->GetStorageShape(), dims, slot)) return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    // Conservatively require room for the declared UB AND the separate cache/reserve.
    if (cores == 0 || ubBytes < kDynamicUbBytes+kMinDCacheBytes+kCompilerReservedBytes)
        return ge::GRAPH_FAILED;
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(cores, static_cast<uint64_t>(dims.queries)*dims.shards));
    optiling::Bm25FusedScoreTopKTilingData data;
    data.set_queryCount(dims.queries); data.set_docCount(dims.docs); data.set_vocabSize(dims.vocab);
    data.set_postingCount(dims.postings); data.set_queryStorageCount(dims.queryStorage);
    data.set_shards(dims.shards); data.set_k(dims.k); data.set_debug(dims.debug); data.set_blockNum(blocks);
    auto *workspace = context->GetWorkspaceSizes(1); auto *raw = context->GetRawTilingData();
    const uint64_t systemBytes = platform.GetLibApiWorkSpaceSize();
    if (workspace == nullptr || raw->GetData() == nullptr || data.GetDataSize() > raw->GetCapacity() ||
        systemBytes > std::numeric_limits<std::size_t>::max()) return ge::GRAPH_FAILED;
    workspace[0] = static_cast<std::size_t>(systemBytes); // No expanded TF or full-score GM workspace.
    context->SetBlockDim(blocks);
    if (context->SetLocalMemorySize(kDynamicUbBytes) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity()); raw->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Bm25FusedScoreTopK : public OpDef {
public:
    explicit Bm25FusedScoreTopK(const char *name) : OpDef(name)
    {
        this->Input("term_offsets").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("posting_doc_ids").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("posting_tf").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("doc_norm").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("idf").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("query_offsets").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("query_term_ids").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("partial_scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("partial_ids").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("debug_scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("k").AttrType(REQUIRED).Int();
        this->Attr("shards").AttrType(REQUIRED).Int();
        this->Attr("debug").AttrType(REQUIRED).Int();
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(Tiling); this->AICore().AddConfig("ascend950");
    }
};
OP_ADD(Bm25FusedScoreTopK);
}
