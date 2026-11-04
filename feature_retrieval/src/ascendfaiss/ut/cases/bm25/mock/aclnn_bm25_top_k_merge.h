#ifndef BM25_TEST_MERGE_API_H
#define BM25_TEST_MERGE_API_H
#include "acl/acl_base.h"
#include "aclnn/acl_meta.h"
aclnnStatus aclnnBm25TopKMergeGetWorkspaceSize(const aclTensor *, const aclTensor *, int64_t, int64_t,
                                               const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
aclnnStatus aclnnBm25TopKMerge(void *, uint64_t, aclOpExecutor *, aclrtStream);
#endif // BM25_TEST_MERGE_API_H
