# BM25 Ascend 950 最终交付版

源码同步使用本仓库 `main` 分支；A5 首次克隆到新目录，后续 `git pull --ff-only`。步骤见 [Git 同步说明](docs/GIT_SYNC.md)，原有数据和 vendor 继续保留。

最终实现：**Warp Append评分/局部Top-K＋原32路分级Merge＋常驻索引/Graph完整Search**。Merge投票、自适应清零、未保留流水、旧算子和历史生成链已移除。

当前源码已开始按 IndexSDK 分层整改。结构说明和剩余接入工作见 [IndexSDK 整改说明](docs/INDEXSDK_REFACTOR.md)。下列历史设备结果仅对应整改前版本；当前版本需要重新进行 A5 验收。

已提供可选 `bm25_indexsdk` 主机目标、`AscendIndexBm25` 公开接口及借用 SDK 上下文的资源模式；构建、调用和生命周期要求见 [IndexSDK 接入说明](docs/INDEXSDK_INTEGRATION.md)。本机已完成接口/资源所有权 UT、可执行 clang-tidy 本地配置检查和 Common 安装包搬迁消费验证；实际 CANN/SDK 链接与 A5 验收仍待目标环境执行。静态门禁与安装验证见 [检查说明](docs/STATIC_CHECKS.md)。

目录前缀为 `feature_retrieval/src/ascendfaiss/`：

| 位置 | 内容 |
|---|---|
| common/bm25/ | 可移植查询包、叶块计划、分区目录、Top-K 辅助 |
| ascendhost/include/index/ | SDK 风格的公开 BM25 接口 |
| ascendhost/src/bm25/ | ACL 资源、算子 API、Graph 与完整 Search 实现 |
| ops/ascendc/bm25/{fused,merge}/ | CANN 算子 op_host/op_kernel 与描述文件 |
| test/ascendhost/bm25/ | 基准程序、设备检查与 CPU 参考实现 |
| ut/cases/bm25/ | CPU 正确性测试 |

根目录 `standalone/` 保留 Python 命令入口；`source_lock.json` 记录当前源码指纹；`evidence/delivery_lock_20261029.json` 保存历史验收版本指纹。

当前整改版的目标环境验证与日志回传使用 [A5 两阶段验收](docs/A5_ACCEPTANCE.md)。以下实测仍为历史版本记录。

## 已有实测

固定合成语料、Q8、每查询3词、K10、FP32和严格排名。完整Search从CPU已有词ID到CPU拿到Top-K，包括每次NPU传输和同步；排除建库、首次整库上传/JIT/预热。

- 合理C++ CPU：百万、419万、千万篇已稳定领先。千万篇原会话NPU A2.00277/B1.98027ms，较快CPU A5.06710/B5.06895ms，约2.53/2.56倍；26万篇尚未稳定赢合理CPU，小库CPU边界保留。
- 固定原版BM25S原生接口：七档1024至千万篇均A/B六轮全胜且P95更低；千万篇同期约19.10/19.35倍。大库非线性倍率只限该配置，严格适配开销另报。
- Merge投票：26万篇汇总略慢0.11%/0.34%，未保留。

最终交付新主机入口已于2026-10-29在A5验收：54次选择检查、36次合并检查及千万篇六轮完整Search完成。同会话按每轮无目录/目录32线程CPU较快者比较，A/B均6/6胜出，P95也逐轮更低；六轮中位数汇总CPU为4.05588/4.06112ms，NPU为1.99893/1.97984ms，加速2.029/2.051倍。此次CPU更快，不能沿用旧会话2.53/2.56倍描述新结果；NPU仍约2ms。记录见evidence/final_delivery_device_evidence_20261029.json。

证据来自用户A5黄区终端摘要复核，不是本地独立硬件复现。暂无MT认可真实业务数据、所有CPU/查询条件领先或950用满的结论。

## 使用

1. 运行 `python3 verify_package.py`。
2. 按[构建与验收](docs/BUILD_AND_TEST.md)复用最终安装或从实际CANN模板构建。
3. 用large_corpus准备数据，benchmark_final做六轮对照。
4. 架构、清理范围与边界见[交付说明](docs/FINAL_DELIVERY.md)。

以下为 2026-10-29 整改前交付记录，当时六份算子源码及18份核心文件字节未变。新根CMake已在用户A5编译5个主机程序、CPU自检1/1通过，旧模板两套算子包构建成功；新主机与既有最终安装的设备检查及千万篇完整Search已通过。新构建的两套算子包未重新安装，本次不为它们单独背书。已有安装只走复用分支，不重复生成模板或覆盖安装。
