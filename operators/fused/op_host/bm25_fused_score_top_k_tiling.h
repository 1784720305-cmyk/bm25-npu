#ifndef BM25_FUSED_SCORE_TOP_K_TILING_H
#define BM25_FUSED_SCORE_TOP_K_TILING_H
#include <cstdint>
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(Bm25FusedScoreTopKTilingData)
TILING_DATA_FIELD_DEF(uint32_t, queryCount);
TILING_DATA_FIELD_DEF(uint32_t, docCount);
TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
TILING_DATA_FIELD_DEF(uint32_t, postingCount);
TILING_DATA_FIELD_DEF(uint32_t, queryStorageCount);
TILING_DATA_FIELD_DEF(uint32_t, shards);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, debug);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Bm25FusedScoreTopK, Bm25FusedScoreTopKTilingData)
}
#endif
