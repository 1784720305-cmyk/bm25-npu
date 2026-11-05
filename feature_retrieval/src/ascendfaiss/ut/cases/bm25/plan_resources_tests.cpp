#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "ascendhost/src/bm25/fused_runtime.h"
#include "ascendhost/src/bm25/planned_search.h"
#include "common/bm25/validation.h"
#include <cstdint>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace {
std::string failing;
std::set<void *> freedHost, freedBuffer, destroyedTensor;
int destroyedModel = 0;
int forbidden = 0;
int allocationCalls = 0;
int allocationPolicy = -1;
aclError allocationStatus = 0;
size_t allocationBytes = 0;
aclrtContext current = reinterpret_cast<void *>(1);
aclError Destroy(void *handle, std::set<void *> &seen, const char *stage)
{
    if (failing == stage) {
        return 42;
    }
    ascend::bm25::Check(seen.insert(handle).second, "Handle destroyed twice after cleanup retry");
    return 0;
}
} // namespace

aclError aclInit(const char *)
{
    ++forbidden;
    return 0;
}
aclError aclFinalize()
{
    ++forbidden;
    return 0;
}
aclError aclrtSetDevice(int32_t)
{
    ++forbidden;
    return 0;
}
aclError aclrtResetDevice(int32_t)
{
    ++forbidden;
    return 0;
}
aclError aclrtDestroyContext(aclrtContext)
{
    ++forbidden;
    return 0;
}
aclError aclrtCreateContext(aclrtContext *, int32_t)
{
    ++forbidden;
    return 0;
}
aclError aclrtGetDevice(int32_t *device)
{
    *device = 3;
    return 0;
}
aclError aclrtGetCurrentContext(aclrtContext *result)
{
    *result = current;
    return 0;
}
aclError aclrtCreateStream(aclrtStream *stream)
{
    *stream = reinterpret_cast<void *>(2);
    return 0;
}
aclError aclrtDestroyStream(aclrtStream) { return failing == "destroyStream" ? 42 : 0; }
aclError aclrtSynchronizeStream(aclrtStream) { return failing == "sync" ? 42 : 0; }
const char *aclGetRecentErrMsg() { return "injected failure"; }
aclError aclrtMalloc(void **p, size_t bytes, int policy)
{
    ++allocationCalls;
    allocationBytes = bytes;
    allocationPolicy = policy;
    *p = allocationStatus ? nullptr : reinterpret_cast<void *>(6);
    return allocationStatus;
}
aclError aclrtFreeHost(void *p) { return Destroy(p, freedHost, "freeHost"); }
aclError aclrtFree(void *p) { return Destroy(p, freedBuffer, "freeBuffer"); }
aclError aclDestroyTensor(aclTensor *p) { return Destroy(p, destroyedTensor, "destroyTensor"); }
aclError aclrtMemcpy(void *, size_t, const void *, size_t, int) { return 0; }
aclError aclmdlRICaptureEnd(aclrtStream, aclmdlRI *model)
{
    *model = reinterpret_cast<void *>(3);
    return failing == "captureEnd" ? 42 : 0;
}
aclError aclmdlRIDestroy(aclmdlRI)
{
    if (failing == "destroyModel") {
        return 42;
    }
    ++destroyedModel;
    return 0;
}

int main()
{
    using ascend::bm25::Check;
    try {
        for (const char *stage :
             {"captureEnd", "sync", "destroyModel", "freeHost", "destroyTensor", "freeBuffer", "destroyStream"}) {
            freedHost.clear();
            freedBuffer.clear();
            destroyedTensor.clear();
            destroyedModel = 0;
            bm25search::PlanResources plan;
            auto owner = std::make_shared<int>(1);
            std::weak_ptr<int> weak = owner;
            plan.runtime.Initialize(3, {current, owner});
            owner.reset();
            plan.buffers.stream = plan.tensors.stream = plan.runtime.stream;
            plan.pinned = {reinterpret_cast<void *>(4), reinterpret_cast<void *>(5)};
            plan.buffers.items = {reinterpret_cast<void *>(6), reinterpret_cast<void *>(7)};
            plan.tensors.items = {reinterpret_cast<aclTensor *>(8), reinterpret_cast<aclTensor *>(9)};
            plan.capturing = std::string(stage) == "captureEnd";
            plan.modelValid = !plan.capturing;
            plan.model = reinterpret_cast<void *>(3);
            failing = stage;
            Check(!plan.Close() && !plan.closed && !weak.expired(), "Cleanup failure lost resources/SDK lease");
            failing.clear();
            Check(plan.Close() && plan.closed && weak.expired(), "Plan cleanup could not be retried");
            Check(plan.Close(), "Plan Close must be idempotent");
            Check(freedHost.size() == 2 && freedBuffer.size() == 2 && destroyedTensor.size() == 2 &&
                      destroyedModel == 1 && forbidden == 0,
                  "Cleanup skipped or repeated resource destruction");
        }
        freedBuffer.clear();
        {
            ascend::bm25::Buffers buffers;
            Check(buffers.Allocate(0) == nullptr && allocationCalls == 0, "Zero bytes must not call allocator");
            constexpr size_t largeBytes = 14051296472ULL;
            allocationStatus = 207001;
            bool rejected = false;
            try {
                buffers.Allocate(largeBytes);
            } catch (const std::exception &error) {
                const std::string message = error.what();
                rejected = message.find(std::to_string(largeBytes)) != std::string::npos &&
                           message.find("ACL_MEM_MALLOC_HUGE_FIRST") != std::string::npos &&
                           message.find("207001") != std::string::npos;
            }
            Check(rejected && buffers.items.empty() && freedBuffer.empty(),
                  "Allocation failure must preserve bytes/policy/code and register no handle");
            allocationStatus = 0;
            Check(buffers.Allocate(largeBytes) == reinterpret_cast<void *>(6) && allocationBytes == largeBytes &&
                      allocationPolicy == ACL_MEM_MALLOC_HUGE_FIRST && allocationCalls == 2 &&
                      buffers.items.size() == 1,
                  "Large allocation lost size, policy or ownership");
            Check(buffers.Close() && buffers.Close() && freedBuffer.size() == 1,
                  "Successful large allocation must be freed exactly once");
        }
        std::cout << "PASS: seven teardown failures and large-allocation size/policy/error/ownership contracts\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
