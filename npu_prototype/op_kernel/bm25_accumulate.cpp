// Each group has one writer: no atomic float addition or cross-block synchronization.
// Not yet compiled or executed on A5.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/device_warp_functions.h"

namespace {
constexpr uint32_t kThreads = 256;
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void AccumulateSimt(
    __gm__ const float *values, __gm__ const int64_t *offsets, __gm__ float *scores,
    uint32_t groupCount, uint64_t valueCount, uint32_t blockId, uint32_t blockNum)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    const uint64_t stride = static_cast<uint64_t>(blockNum) * kThreads;
    for (uint64_t group = static_cast<uint64_t>(blockId) * kThreads + thread;
         group < groupCount; group += stride) {
        const int64_t begin = offsets[group];
        const int64_t end = offsets[group + 1];
        // Host must validate monotonic offsets. Bounds guard prevents out-of-range reads.
        if (begin < 0 || end < begin || static_cast<uint64_t>(end) > valueCount) {
            scores[group] = -1.0F; // Nonnegative BM25 contract makes this a detectable invalid value.
            continue;
        }
        float total = 0.0F;
        for (int64_t i = begin; i < end; ++i) total += values[i];
        scores[group] = total; // Empty groups also overwrite their output with zero.
    }
}
}

extern "C" __global__ __aicore__ void bm25_accumulate(
    GM_ADDR values, GM_ADDR offsets, GM_ADDR scores, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    AscendC::Simt::VF_CALL<AccumulateSimt>(
        AscendC::Simt::Dim3{kThreads, 1, 1}, reinterpret_cast<__gm__ const float *>(values),
        reinterpret_cast<__gm__ const int64_t *>(offsets), reinterpret_cast<__gm__ float *>(scores),
        data.groupCount, data.valueCount, static_cast<uint32_t>(AscendC::GetBlockIdx()), data.blockNum);
}
