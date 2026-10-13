// Follow IndexSDK's A5 SIMT VF_CALL structure; own BM25 arithmetic.
// Draft only: needs CANN compilation and A5 device validation.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/device_warp_functions.h"

namespace {
constexpr uint32_t kThreads = 256;
constexpr float kK1 = 1.2F;

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void ContributionSimt(
    __gm__ const float *tf, __gm__ const float *docNorm, __gm__ const float *idf,
    __gm__ float *output, uint32_t count, uint32_t blockId, uint32_t blockNum)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    const uint64_t stride = static_cast<uint64_t>(blockNum) * kThreads;
    for (uint64_t i = static_cast<uint64_t>(blockId) * kThreads + thread; i < count; i += stride) {
        // Same FP32 expression ordering as the CPU reference; k1 is fixed for this first prototype.
        const float numerator = tf[i] * (kK1 + 1.0F);
        output[i] = (idf[i] * numerator) / (tf[i] + docNorm[i]);
    }
}
}

extern "C" __global__ __aicore__ void bm25_contribution(
    GM_ADDR tf, GM_ADDR docNorm, GM_ADDR idf, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    AscendC::Simt::VF_CALL<ContributionSimt>(
        AscendC::Simt::Dim3{kThreads, 1, 1}, reinterpret_cast<__gm__ const float *>(tf),
        reinterpret_cast<__gm__ const float *>(docNorm), reinterpret_cast<__gm__ const float *>(idf),
        reinterpret_cast<__gm__ float *>(output), data.recordCount, static_cast<uint32_t>(AscendC::GetBlockIdx()),
        data.blockNum);
}
