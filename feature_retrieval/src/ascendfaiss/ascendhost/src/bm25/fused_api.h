#ifndef BM25_ASCENDHOST_SRC_BM25_FUSED_API_H
#define BM25_ASCENDHOST_SRC_BM25_FUSED_API_H
#include "ascendhost/src/bm25/fused_runtime.h"
namespace ascend::bm25 {
struct SearchFusedApi {
    ApiLibrary fused, merge;
    FusedPrepare prepareFused;
    MergePrepare prepareMerge;
    Execute executeFused, executeMerge;
    explicit SearchFusedApi(const fs::path &vendors)
        : fused(vendors / "bm25_fused_score_topk"), merge(vendors / "bm25_topk_merge")
    {
        prepareFused = fused.Symbol<FusedPrepare>("aclnnBm25FusedScoreTopKGetWorkspaceSize");
        executeFused = fused.Symbol<Execute>("aclnnBm25FusedScoreTopK");
        prepareMerge = merge.Symbol<MergePrepare>("aclnnBm25TopKMergeGetWorkspaceSize");
        executeMerge = merge.Symbol<Execute>("aclnnBm25TopKMerge");
    }
};

} // namespace ascend::bm25

#endif // BM25_ASCENDHOST_SRC_BM25_FUSED_API_H
