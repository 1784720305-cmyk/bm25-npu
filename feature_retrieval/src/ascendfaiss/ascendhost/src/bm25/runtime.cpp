#include "ascendhost/src/bm25/runtime.h"
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include "ascendhost/include/impl/Bm25RuntimeBinding.h"
#include "common/bm25/error.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <memory>
#include <stdexcept>

namespace ascend::bm25 {
void RequireAcl(aclError code, const char *operation)
{
    if (code == ACL_ERROR_NONE) {
        return;
    }
    const char *detail = aclGetRecentErrMsg();
    throw DeviceError(code, std::string(operation) + " failed, code=" + std::to_string(code) +
                                ", detail=" + (detail ? detail : "<none>"));
}

void Runtime::Initialize(int deviceId, const RuntimeBinding &binding)
{
    CheckArgument(!context && !stream && !initialized && !selected && !ownerLease_, "Runtime already initialized");
    CheckArgument(deviceId >= 0, "Invalid device ID");
    CheckArgument(static_cast<bool>(binding.context) == static_cast<bool>(binding.owner),
                  "External context and resource owner must be supplied together");
    borrowedContext_ = false;
    device = deviceId;
    if (binding.context) {
        aclrtContext current = nullptr;
        int32_t currentDevice = -1;
        RequireAcl(aclrtGetCurrentContext(&current), "get SDK context");
        RequireAcl(aclrtGetDevice(&currentDevice), "get SDK device");
        CheckArgument(current == binding.context && currentDevice == device, "SDK context/device is not current");
        ownerLease_ = std::make_unique<std::shared_ptr<void>>(binding.owner);
        borrowedContext_ = true;
        context = binding.context;
    } else {
        RequireAcl(aclInit(nullptr), "aclInit");
        initialized = true;
        RequireAcl(aclrtSetDevice(device), "set device");
        selected = true;
        aclrtContext created = nullptr;
        RequireAcl(aclrtCreateContext(&created, device), "create context");
        if (!created) {
            throw std::runtime_error("ACL returned a null context");
        }
        context = created;
    }
    // A dedicated stream keeps graph capture off the SDK's shared default stream.
    aclrtStream createdStream = nullptr;
    RequireAcl(aclrtCreateStream(&createdStream), "create BM25 stream");
    if (!createdStream) {
        throw std::runtime_error("ACL returned a null stream");
    }
    stream = createdStream;
}

bool Runtime::HasCurrentContext() const noexcept
{
    if (!borrowedContext_ || !context) {
        return true;
    }
    aclrtContext current = nullptr;
    return aclrtGetCurrentContext(&current) == ACL_ERROR_NONE && current == context;
}

void Runtime::RequireCurrentContext() const
{
    CheckArgument(HasCurrentContext(), "Activate the bound SDK context before calling BM25");
}

bool Runtime::Close() noexcept
{
    if (!HasCurrentContext()) {
        return false;
    }
    if (stream) {
        if (aclrtSynchronizeStream(stream) != ACL_ERROR_NONE || aclrtDestroyStream(stream) != ACL_ERROR_NONE) {
            return false;
        }
        stream = nullptr;
    }
    if (context && !borrowedContext_) {
        if (aclrtDestroyContext(context) != ACL_ERROR_NONE) {
            return false;
        }
    }
    context = nullptr;
    if (selected) {
        if (aclrtResetDevice(device) != ACL_ERROR_NONE) {
            return false;
        }
        selected = false;
    }
    if (initialized) {
        if (aclFinalize() != ACL_ERROR_NONE) {
            return false;
        }
        initialized = false;
    }
    ownerLease_.reset();
    return true;
}

void Runtime::Abandon() noexcept
{
    // The SDK proxy may reset the device when its last reference is released.
    // Preserve that reference too when a running graph cannot be safely destroyed.
    (void)ownerLease_.release();
    stream = nullptr;
    context = nullptr;
    initialized = false;
    selected = false;
}

Runtime::~Runtime()
{
    if (!Close()) {
        std::cerr << "cleanup_failed=BM25 runtime retained until process exit\n";
        Abandon();
    }
}
} // namespace ascend::bm25
