# BM25 Ascend 950 最终交付版

最终实现：**Warp Append评分/局部Top-K＋原32路分级Merge＋常驻索引/Graph完整Search**。Merge投票、自适应清零、未保留流水、旧算子和历史生成链已移除。

| 位置 | 内容 |
|---|---|
| operators/fused/ | 最终评分/局部选择的直接源码，不再逐层生成 |
| operators/merge/ | 最终分级合并，无投票快路 |
| common/ | 查询包、叶块计划、全词分区目录 |
| npu_prototype/large_search_host/ | 最终完整Search入口 |
| 其余保留的npu_prototype文件 | 复用的资源/Graph/API/计时/设备检查支持；保留路径以维持已测核心文件字节 |
| cpu_reference/ | 已验收无目录/目录C++ CPU及严格Top-K |
| standalone/benchmark_final.py | 唯一完整Search基准入口 |
| reference_only/bm25s_original/ | 未修改的固定原版BM25S，可选对手 |
| evidence/ | 最终Warp Append、BM25S与Merge否决证据 |
| delivery_lock.json | 最终内核导出及核心主机字节指纹 |

## 已有实测

固定合成语料、Q8、每查询3词、K10、FP32和严格排名。完整Search从CPU已有词ID到CPU拿到Top-K，包括每次NPU传输和同步；排除建库、首次整库上传/JIT/预热。

- 合理C++ CPU：百万、419万、千万篇已稳定领先。千万篇原会话NPU A2.00277/B1.98027ms，较快CPU A5.06710/B5.06895ms，约2.53/2.56倍；26万篇尚未稳定赢合理CPU，小库CPU边界保留。
- 固定原版BM25S原生接口：七档1024至千万篇均A/B六轮全胜且P95更低；千万篇同期约19.10/19.35倍。大库非线性倍率只限该配置，严格适配开销另报。
- Merge投票：26万篇汇总略慢0.11%/0.34%，未保留。

证据来自用户A5黄区终端摘要复核，不是本地独立硬件复现。暂无MT认可真实业务数据、所有CPU/查询条件领先或950用满的结论。

## 使用

1. 运行 `python3 verify_package.py`。
2. 按[构建与验收](docs/BUILD_AND_TEST.md)复用最终安装或从实际CANN模板构建。
3. 用large_corpus准备数据，benchmark_final做六轮对照。
4. 架构、清理范围与边界见[交付说明](docs/FINAL_DELIVERY.md)。

清理只固化/裁剪依赖，六份算子源码及18份核心文件字节未变。新构建/编排入口经过本地检查，但本机没有A5/CANN，未生成新的设备性能结论。复用已有安装不重跑全部旧实验；全新构建需黄区设备检查。
