#ifndef BM25_TEST_FUSED_API_H
#define BM25_TEST_FUSED_API_H
#include "acl/acl.h"
aclnnStatus aclnnBm25FusedScoreTopKGetWorkspaceSize(const aclTensor *, const aclTensor *, const aclTensor *,
                                                    const aclTensor *, const aclTensor *, const aclTensor *,
                                                    const aclTensor *, int64_t, int64_t, int64_t, const aclTensor *,
                                                    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
aclnnStatus aclnnBm25FusedScoreTopK(void *, uint64_t, aclOpExecutor *, aclrtStream);
#endif // BM25_TEST_FUSED_API_H
