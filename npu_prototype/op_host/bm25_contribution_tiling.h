#ifndef BM25_CONTRIBUTION_TILING_H
#define BM25_CONTRIBUTION_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Bm25ContributionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, recordCount);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Bm25Contribution, Bm25ContributionTilingData)
}
#endif
