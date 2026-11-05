# BM25 Ascend 950 最终交付版

源码同步使用本仓库 `main` 分支；A5 首次克隆到新目录，后续 `git pull --ff-only`。步骤见 [Git 同步说明](docs/GIT_SYNC.md)，原有数据和 vendor 继续保留。

最终实现：**Warp Append评分/局部Top-K＋原32路分级Merge＋常驻索引/Graph完整Search**。Merge投票、自适应清零、未保留流水、旧算子和历史生成链已移除。

本轮 IndexSDK 分层整改与约定的 A5 验收已完成。2026-11-04，代码提交 `9a99bbc` 在真实 CANN 9.2.0 / IndexSDK 环境完成构建、单元测试、生产 clang-tidy、安装搬迁消费、共享 SDK 资源检查和千万篇数据六轮完整 Search，最终状态为 `all_requested_checks_passed`。[验收记录](evidence/indexsdk_a5_acceptance_20261104.json)保存回传报告摘要及本地复核结果；本轮之后的文档更新不改变已验收的生产源码。

已提供可选 `bm25_indexsdk` 主机目标、`AscendIndexBm25` 公开接口及借用 SDK 上下文的资源模式；构建、调用和生命周期要求见 [IndexSDK 接入说明](docs/INDEXSDK_INTEGRATION.md)。真实 A5 检查覆盖 Common/IndexSDK 安装后的调用方、SDK 代理引用与上下文生命周期、A/B 独立 oracle、非法输入恢复和关闭后 SDK 流可用性。规范映射及静态检查范围见 [检查说明](docs/STATIC_CHECKS.md)，整改过程见 [整改说明](docs/INDEXSDK_REFACTOR.md)。

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

当前整改版的复现命令和必要环境变量见 [A5 两阶段验收](docs/A5_ACCEPTANCE.md)。本轮千万篇六轮中位数汇总：NPU A/B 为 **2.004 / 1.979 ms**，每轮均快于同轮两种 32 线程 C++ CPU 实现中的较快者，P95 也更低。本次 CPU 波动较大，本会话约 6.05 / 6.97 倍的倍率不表示结构整改带来了对应的 NPU 优化，也不替代历史会话结果。后续“已有实测”保留整改前的记录。

CPU 相对旧版的速度保持尚未由上述门槛证明。[CPU 回归核查](docs/CPU_REGRESSION.md)已收到同机绑定后的三组对照：旧版 A/B 为 5.473/5.552 ms，新版 5.753/5.785 ms，新源码恢复旧参数后为 5.395/5.390 ms。此前数倍差距未复现；同一新版两次试验自身相差约 13%，且存在较高系统负载背景，不能把汇总约 4%～5% 的差异认定为编译参数的固定开销。生产 C++、算子及当前编译策略保持不变，剩余差异待空闲窗口复核。

2026-11-05 的 [原版 BM25S 性能回归](docs/BM25S_PERFORMANCE.md)已完成：千万篇/Q8/K10 的 NPU A/B 为 **1.993/1.979 ms**，较历史分别变化 **-0.51%/+0.04%**，满足“与此前接近”的目标。原版 BM25S 同轮较快配置约 31.383/31.446 ms，NPU 延迟领先约 15.75/15.89 倍，A/B 均六轮全胜；倍率下降来自本会话 CPU 更快，NPU 仍约 2 ms。[回传证据](evidence/bm25s_regression_a5_20261105.json)已复核。下一步使用 [SciFact 公开真实数据](docs/SCIFACT_REAL_DATA.md)，本地导入/CPU 校验已完成，真实 A5 设备结果待运行。C++ 配对保留作辅助诊断；若低干扰下稳定慢 4%～5%，仍作为实测开销记录。

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
