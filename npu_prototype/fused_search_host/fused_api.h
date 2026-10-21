#pragma once
struct SearchFusedApi {
    ApiLibrary fused, merge;
    FusedPrepare prepareFused;
    MergePrepare prepareMerge;
    Execute executeFused, executeMerge;
    explicit SearchFusedApi(const fs::path &vendors)
        : fused(vendors/"bm25_fused_score_topk"), merge(vendors/"bm25_topk_merge")
    {
        prepareFused = fused.Symbol<FusedPrepare>("aclnnBm25FusedScoreTopKGetWorkspaceSize");
        executeFused = fused.Symbol<Execute>("aclnnBm25FusedScoreTopK");
        prepareMerge = merge.Symbol<MergePrepare>("aclnnBm25TopKMergeGetWorkspaceSize");
        executeMerge = merge.Symbol<Execute>("aclnnBm25TopKMerge");
    }
};

