#ifndef BM25_ACCUMULATE_TILING_H
#define BM25_ACCUMULATE_TILING_H
#include <cstdint>
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(Bm25AccumulateTilingData)
TILING_DATA_FIELD_DEF(uint32_t, groupCount);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint64_t, valueCount);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Bm25Accumulate, Bm25AccumulateTilingData)
}
#endif
