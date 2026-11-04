#ifndef BM25_TEST_ACL_MOCK_H
#define BM25_TEST_ACL_MOCK_H
#include <cstdint>
#include <cstddef>
using aclError = int;
using aclrtContext = void *;
using aclrtStream = void *;
constexpr aclError ACL_ERROR_NONE = 0;
aclError aclInit(const char *config);
aclError aclFinalize();
aclError aclrtSetDevice(int32_t device);
aclError aclrtResetDevice(int32_t device);
aclError aclrtGetDevice(int32_t *device);
aclError aclrtGetCurrentContext(aclrtContext *context);
aclError aclrtCreateContext(aclrtContext *context, int32_t device);
aclError aclrtDestroyContext(aclrtContext context);
aclError aclrtCreateStream(aclrtStream *stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtSynchronizeStream(aclrtStream stream);
const char *aclGetRecentErrMsg();
struct aclTensor {};
struct aclOpExecutor {};
using aclmdlRI = void *;
using aclDataType = int;
using aclnnStatus = int;
constexpr int ACL_FORMAT_ND = 0;
constexpr int ACL_FLOAT = 1;
constexpr int ACL_INT32 = 2;
constexpr int ACL_INT64 = 3;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;
aclTensor *aclCreateTensor(const int64_t *, uint64_t, aclDataType, const int64_t *, int64_t, int, const int64_t *,
                           uint64_t, void *);
aclError aclDestroyTensor(aclTensor *);
aclError aclrtMalloc(void **, size_t, int);
aclError aclrtMallocHost(void **, size_t);
aclError aclrtFree(void *);
aclError aclrtFreeHost(void *);
aclError aclrtMemcpy(void *, size_t, const void *, size_t, int);
aclError aclmdlRICaptureEnd(aclrtStream, aclmdlRI *);
aclError aclmdlRIDestroy(aclmdlRI);
#endif // BM25_TEST_ACL_MOCK_H
