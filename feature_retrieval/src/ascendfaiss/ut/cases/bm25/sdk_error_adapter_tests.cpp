#include "ascendhost/include/impl/Bm25ErrorAdapter.h"
#include "common/bm25/error_info.h"
#include "common/ErrorCode.h"
#include "common/bm25/error.h"
#include "common/bm25/validation.h"
#include <cstring>
#include <exception>
#include <iostream>

int main()
{
    using ascend::APP_ERR_OK;
    using ascend::APP_ERR_INVALID_PARAM;
    using ascend::APP_ERR_ILLEGAL_OPERATION;
    using ascend::APP_ERR_INNER_ERROR;
    using ascend::bm25::ErrorInfo;
    using ascend::bm25::Check;
    using ascend::bm25::InvokeSdkOperation;
    using ascend::bm25::InvalidArgument;
    using ascend::bm25::InvalidState;
    using ascend::bm25::DeviceError;
    try {
        ErrorInfo info;
        Check(InvokeSdkOperation([] {}, &info) == APP_ERR_OK, "Success mapping failed");
        Check(InvokeSdkOperation([] { throw InvalidArgument("query"); }, &info) == APP_ERR_INVALID_PARAM &&
                  std::string(info.message.data()) == "query",
              "Input error mapping failed");
        Check(InvokeSdkOperation([] { throw InvalidState("closed"); }, &info) == APP_ERR_ILLEGAL_OPERATION,
              "State error mapping failed");
        Check(InvokeSdkOperation([] { throw DeviceError(161001, "prepare"); }, &info) == APP_ERR_INNER_ERROR &&
                  info.aclCode == 161001 && std::string(info.message.data()) == "prepare",
              "ACL diagnostic lost");
        Check(InvokeSdkOperation([] { throw 42; }, &info) == APP_ERR_INNER_ERROR && info.aclCode == 0,
              "Unknown exception escaped SDK boundary");
        const std::string longMessage(2048, 'x');
        Check(InvokeSdkOperation([&] { throw InvalidArgument(longMessage); }, &info) == APP_ERR_INVALID_PARAM &&
                  info.message.back() == '\0' && std::strlen(info.message.data()) == 1023,
              "Diagnostic truncation failed");
        Check(InvokeSdkOperation([] {}, &info) == APP_ERR_OK && info.message[0] == '\0' && info.aclCode == 0,
              "Old error leaked into success");
        std::cout << "PASS: actual SDK error constants and retained diagnostics\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
