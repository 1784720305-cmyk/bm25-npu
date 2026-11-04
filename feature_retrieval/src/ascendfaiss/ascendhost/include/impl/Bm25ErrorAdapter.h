#ifndef BM25_ERROR_ADAPTER_H
#define BM25_ERROR_ADAPTER_H
#include "common/ErrorCode.h"
#include "common/bm25/error.h"
#include "common/bm25/error_info.h"
#include <utility>

namespace ascend::bm25 {
// SDK integration boundary. Uses IndexSDK's real error constants; no local numeric copies.
// DeviceError retains the raw ACL code for diagnostics inside the supplied operation.
template <class Operation> APP_ERROR InvokeSdkOperation(Operation &&operation, ErrorInfo *error = nullptr) noexcept
{
    if (error) {
        error->Set(nullptr);
    }
    try {
        std::forward<Operation>(operation)();
        return APP_ERR_OK;
    } catch (const InvalidArgument &failure) {
        if (error) {
            error->Set(failure.what());
        }
        return APP_ERR_INVALID_PARAM;
    } catch (const InvalidState &failure) {
        if (error) {
            error->Set(failure.what());
        }
        return APP_ERR_ILLEGAL_OPERATION;
    } catch (const DeviceError &failure) {
        if (error) {
            error->Set(failure.what(), failure.GetCode());
        }
        return APP_ERR_INNER_ERROR;
    } catch (const std::exception &failure) {
        if (error) {
            error->Set(failure.what());
        }
        return APP_ERR_INNER_ERROR;
    } catch (...) {
        if (error) {
            error->Set("Unknown BM25 failure");
        }
        return APP_ERR_INNER_ERROR;
    }
}
} // namespace ascend::bm25
#endif // BM25_ERROR_ADAPTER_H
