#ifndef BM25_WARP_TOP_K_TILING_H
#define BM25_WARP_TOP_K_TILING_H
#include <cstdint>
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(Bm25TopKTilingData)
TILING_DATA_FIELD_DEF(uint32_t, queryCount);
TILING_DATA_FIELD_DEF(uint32_t, docCount);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, partitionCount);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Bm25TopK, Bm25TopKTilingData)
}
#endif
