#ifndef BM25_TEST_ACL_META_MOCK_H
#define BM25_TEST_ACL_META_MOCK_H
#include "acl/acl_base_rt.h"
#include <cstdint>
struct aclTensor {};
struct aclOpExecutor {};
using aclnnStatus = int;
aclTensor *aclCreateTensor(const int64_t *, uint64_t, aclDataType, const int64_t *, int64_t, int, const int64_t *,
                           uint64_t, void *);
aclError aclDestroyTensor(aclTensor *);
#endif // BM25_TEST_ACL_META_MOCK_H
