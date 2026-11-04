#include "acl/acl.h"
#include "ascendhost/src/bm25/runtime.h"
#include "common/bm25/error.h"
#include "common/bm25/validation.h"
#include <cstdint>
#include <initializer_list>
#include <exception>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

namespace {
std::vector<std::string> calls;
aclrtContext current = reinterpret_cast<void *>(1);
int32_t device = 3;
std::string failing;
int Call(const char *name)
{
    calls.emplace_back(name);
    return failing == name ? 42 : 0;
}
void Reset()
{
    calls.clear();
    failing.clear();
    current = reinterpret_cast<void *>(1);
    device = 3;
}
void Expect(std::initializer_list<const char *> wanted)
{
    std::vector<std::string> expected(wanted.begin(), wanted.end());
    ascend::bm25::Check(calls == expected, "Unexpected ACL ownership/cleanup calls");
}
} // namespace

aclError aclInit(const char *) { return Call("init"); }
aclError aclFinalize() { return Call("finalize"); }
aclError aclrtSetDevice(int32_t) { return Call("setDevice"); }
aclError aclrtResetDevice(int32_t) { return Call("resetDevice"); }
aclError aclrtGetDevice(int32_t *result)
{
    *result = device;
    return Call("getDevice");
}
aclError aclrtGetCurrentContext(aclrtContext *result)
{
    *result = current;
    return Call("getContext");
}
aclError aclrtCreateContext(aclrtContext *result, int32_t)
{
    *result = current;
    return Call("createContext");
}
aclError aclrtDestroyContext(aclrtContext) { return Call("destroyContext"); }
aclError aclrtCreateStream(aclrtStream *result)
{
    const auto code = Call("createStream");
    if (!code) {
        *result = reinterpret_cast<void *>(2);
    }
    return code;
}
aclError aclrtDestroyStream(aclrtStream) { return Call("destroyStream"); }
aclError aclrtSynchronizeStream(aclrtStream) { return Call("sync"); }
const char *aclGetRecentErrMsg() { return "injected failure"; }

int main()
{
    using ascend::bm25::Runtime;
    using ascend::bm25::Check;
    using ascend::bm25::InvalidArgument;
    using ascend::bm25::DeviceError;
    try {
        Reset();
        {
            Runtime runtime;
            runtime.Initialize(3);
            Check(runtime.Close(), "Standalone close failed");
        }
        Expect({"init", "setDevice", "createContext", "createStream", "sync", "destroyStream", "destroyContext",
                "resetDevice", "finalize"});
        Reset();
        {
            auto owner = std::make_shared<int>(7);
            std::weak_ptr<int> weak = owner;
            Runtime runtime;
            runtime.Initialize(3, {current, owner});
            owner.reset();
            Check(!weak.expired(), "SDK owner released while runtime is active");
            current = reinterpret_cast<void *>(9);
            bool rejected = false;
            try {
                runtime.RequireCurrentContext();
            } catch (const InvalidArgument &) {
                rejected = true;
            }
            Check(rejected && !runtime.Close() && !weak.expired(), "Wrong-context cleanup must retain resources");
            current = reinterpret_cast<void *>(1);
            failing = "sync";
            Check(!runtime.Close() && !weak.expired(), "Failed sync released SDK owner");
            failing.clear();
            Check(runtime.Close() && weak.expired(), "External close failed to release owner");
            calls.erase(calls.begin(), calls.end() - 3);
            Expect({"getContext", "sync", "destroyStream"});
        }
        // External initialization and teardown never own process/device/context lifetime.
        for (const char *failure : {"createStream", "destroyStream"}) {
            Reset();
            {
                Runtime runtime;
                auto owner = std::make_shared<int>(1);
                if (std::string(failure) == "createStream") {
                    failing = failure;
                }
                try {
                    runtime.Initialize(3, {current, owner});
                } catch (const DeviceError &error) {
                    Check(error.GetCode() == 42, "ACL error code lost");
                }
                if (std::string(failure) == "destroyStream") {
                    failing = failure;
                    Check(!runtime.Close(), "Injected destroy failure ignored");
                }
                failing.clear();
                Check(runtime.Close(), "Partial cleanup could not be retried");
            }
            for (const auto &call : calls) {
                Check(call != "init" && call != "finalize" && call != "resetDevice" && call != "destroyContext" &&
                          call != "setDevice",
                      "Borrowed context was treated as owned");
            }
        }
        Reset();
        {
            Runtime runtime;
            bool rejected = false;
            try {
                runtime.Initialize(4, {current, std::make_shared<int>(1)});
            } catch (const InvalidArgument &) {
                rejected = true;
            }
            Check(rejected, "Mismatched SDK device accepted");
        }
        Expect({"getContext", "getDevice"});
        Reset();
        {
            Runtime runtime;
            runtime.Initialize(3, {current, std::make_shared<int>(1)});
            Check(runtime.Close(), "External close before ownership-mode change failed");
            calls.clear();
            runtime.Initialize(3);
            Check(runtime.Close(), "Standalone close after external use failed");
        }
        Expect({"init", "setDevice", "createContext", "createStream", "sync", "destroyStream", "destroyContext",
                "resetDevice", "finalize"});
        std::cout << "PASS: owned/borrowed ACL lifecycle, context affinity, owner lifetime and failed cleanup retry\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
