#ifndef BM25_TEST_ACL_RT_MOCK_H
#define BM25_TEST_ACL_RT_MOCK_H
#include "acl/acl_base.h"
#include <cstddef>
#include <cstdint>
using aclmdlRI = void *;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 2;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int ACL_HBM_MEM = 0;
constexpr int ACL_MODEL_RI_CAPTURE_MODE_GLOBAL = 0;
aclError aclInit(const char *config);
aclError aclFinalize();
const char *aclGetRecentErrMsg();
aclError aclrtSetDevice(int32_t device);
aclError aclrtResetDevice(int32_t device);
aclError aclrtGetDevice(int32_t *device);
aclError aclrtGetCurrentContext(aclrtContext *context);
aclError aclrtCreateContext(aclrtContext *context, int32_t device);
aclError aclrtDestroyContext(aclrtContext context);
aclError aclrtCreateStream(aclrtStream *stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtMalloc(void **, size_t, int);
aclError aclrtMallocHost(void **, size_t);
aclError aclrtFree(void *);
aclError aclrtFreeHost(void *);
aclError aclrtGetMemInfo(int, size_t *, size_t *);
aclError aclrtMemcpy(void *, size_t, const void *, size_t, int);
aclError aclrtMemcpyAsync(void *, size_t, const void *, size_t, int, aclrtStream);
aclError aclmdlRICaptureBegin(aclrtStream, int);
aclError aclmdlRICaptureEnd(aclrtStream, aclmdlRI *);
aclError aclmdlRIExecuteAsync(aclmdlRI, aclrtStream);
aclError aclmdlRIDestroy(aclmdlRI);
#endif // BM25_TEST_ACL_RT_MOCK_H
