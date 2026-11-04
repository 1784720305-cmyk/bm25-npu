#include "ascendhost/src/bm25/planned_search.h"
#include <array>
#include <cstddef>
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include <algorithm>
#include <iostream>
namespace bm25search {
void PlanResources::CheckGuards() const
{
    runtime.RequireCurrentContext();
    std::array<unsigned char, 64> actual{};
    for (const auto &guard : guards)
        for (size_t offset : {size_t(0), size_t(64) + guard.size}) {
            Require(aclrtMemcpy(actual.data(), 64, guard.allocation + offset, 64, ACL_MEMCPY_DEVICE_TO_HOST),
                    "read plan guard outside timing");
            Check(std::all_of(actual.begin(), actual.end(), [](unsigned char value) { return value == 0xa7; }),
                  "Plan arena guard overwritten");
        }
}

bool PlanResources::Close()
{
    if (closed)
        return true;
    if (!runtime.HasCurrentContext()) {
        return false;
    }
    if (capturing) {
        // CaptureEnd is also required on exceptions; never synchronize an active capture.
        if (aclmdlRICaptureEnd(runtime.stream, &model) != ACL_ERROR_NONE)
            return false;
        capturing = false;
        modelValid = true;
    }
    if (runtime.stream && aclrtSynchronizeStream(runtime.stream) != ACL_ERROR_NONE)
        return false;
    if (modelValid) {
        if (aclmdlRIDestroy(model) != ACL_ERROR_NONE)
            return false;
        modelValid = false;
    }
    while (!pinned.empty()) {
        if (aclrtFreeHost(pinned.back()) != ACL_ERROR_NONE) {
            return false;
        }
        pinned.pop_back();
    }
    tensors.stream = nullptr;
    buffers.stream = nullptr;
    if (!tensors.Close() || !buffers.Close() || !runtime.Close()) {
        return false;
    }
    closed = true;
    return true;
}

PlanResources::~PlanResources()
{
    if (!Close()) {
        // A failed capture/sync/destroy cannot prove the graph has stopped using addresses.
        // Retain ALL ownership until process exit instead of freeing a subset underneath it.
        std::cerr << "cleanup_failed=plan resources retained until process exit\n";
        tensors.items.clear();
        buffers.items.clear();
        pinned.clear();
        tensors.stream = nullptr;
        buffers.stream = nullptr;
        runtime.Abandon();
    }
}

} // namespace bm25search
