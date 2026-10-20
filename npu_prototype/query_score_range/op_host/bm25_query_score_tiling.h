#ifndef BM25_QUERY_SCORE_RANGE_TILING_H
#define BM25_QUERY_SCORE_RANGE_TILING_H
#include <cstdint>
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(Bm25QueryScoreTilingData)
TILING_DATA_FIELD_DEF(uint32_t, queryCount);
TILING_DATA_FIELD_DEF(uint32_t, docCount);
TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
TILING_DATA_FIELD_DEF(uint32_t, postingCount);
TILING_DATA_FIELD_DEF(uint32_t, queryStorageCount);
TILING_DATA_FIELD_DEF(uint32_t, partitionsPerQuery);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Bm25QueryScore, Bm25QueryScoreTilingData)
}
#endif
