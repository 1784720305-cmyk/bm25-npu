# IndexSDK 合入后 A5 验收与交付

2026-11-06，冻结的 `split` 已合入独立 IndexSDK 源码树，并完成真实 CANN 构建、静态检查、安装搬迁消费及实际公开接口的设备验收。执行提交为 `e158fd202064299f08077e96f6674134241df142`，SDK 基线为 `59213c4ca5ebb643c031c8b819e90e3a5cc21a8c`。**本轮工作收尾，停止性能调参，已完成的矩阵无需重跑。**

[验收证据](../evidence/indexsdk_postintegration_a5_20261106.json)归档用户回传的执行命令、六轮中位/P95 数值、完整质量指标、实际 SDK/算子指纹，以及本地独立复算。A5 原始报告、二进制和逐次样本保留在设备上；它们的 SHA-256 来自只读回传，未冒充本机下载重算。

## 已通过的检查

- BM25 的五个实现直接编入原 SDK 的 `ascendfaiss_host`，实际产物为 `libascendfaiss.so`。程序的 `ldd` 指向本次 SDK 构建目录。
- 算子构建安装、SDK 共享库编译、七项 CPU 单元检查通过；七个实际编译单元的 clang-tidy 均通过，未使用 mock，检查配置与 warnings-as-errors 保持。
- 安装后的独立消费程序在搬迁路径编译运行，BM25 公开符号导出检查通过。静态检查恢复过程保留了原 SDK 和两个检查程序的二进制 SHA-256。
- 实际 SDK 资源代理的拒绝查询恢复、两次 Init/Close、引用生命周期，以及 Close 后 SDK 上下文、设备和流仍有效的检查通过。
- 实际 `AscendIndexBm25::Search` 完成三组完整质量检查。设备脚本逐项比较 INT64 labels 对应的严格 ID、FP32 分数位与冻结版，并复现 qrels 指标。
- 六组固定 Q8/K10 输入各完成六轮 SDK/独立主机配对，奇偶轮交换进程先后；共 72 个性能进程、4320 次正式 Search。两者使用同一个从合入源码重建的算子包，独立主机继续执行完整分数、保护区、Merge 树及索引回读诊断。

主机阶段报告的状态仍为 `integrated_IndexSDK_build_static_UT_install_passed_device_pending`，它是主机阶段的历史边界。独立设备报告已达到 `postintegration_actual_public_SDK_quality_and_six_round_Search_checked`；不能仅看前者误判设备仍未测。

## 性能保持情况

计时覆盖调用方进入公开 `Search` 到取得结果，包含接口自身的锁、同步和 INT64 labels 转换。下表为每批 Q8 的六轮中位数汇总，变化相对**同会话、同算子包的独立主机 Search**；负值更快。P95 列比较六轮 P95 的中位数，不是将 180 个样本合并计算的 P95。

| 输入 | SDK A/B ms | 中位延迟变化 A/B | P95 变化 A/B |
|---|---|---|---|
| SciFact | 0.061708 / 0.049039 | +0.77% / +1.20% | +2.97% / +5.22% |
| TREC-COVID | 0.174051 / 0.137436 | +0.75% / +0.13% | +1.69% / +0.46% |
| MS MARCO | 1.946897 / 2.106404 | -0.58% / +0.24% | -0.51% / +0.43% |
| sparse8192 | 0.029632 / 0.030311 | -1.42% / -1.66% | -1.20% / -4.12% |
| sparse65536 | 0.039031 / 0.039592 | -0.47% / -0.97% | -0.08% / -3.85% |
| synthetic10m | 1.765656 / 1.741239 | +0.50% / -0.19% | +0.53% / +0.13% |

在这组固定条件下，中位延迟变化为 -1.66%～+1.20%，符合本轮“接入后与此前性能接近”的目标。**SciFact B 的 P95 +5.22% 保留为实测代价**；千万篇 A 在六轮都略慢，中位增加约 8.70 微秒/Q8，也如实保留。不将这些差异一概归因于系统负载，不宣称零开销或统计等价。

单轮尾延迟也未剔除：SciFact A 第一轮 P95 从 0.063074 ms 增至 0.077747 ms（+23.26%），TREC-COVID A 第二轮从 0.174772 ms 增至 0.192169 ms（+9.95%）。原始六轮汇总均保存在证据中；这些单轮增幅与表中六轮 P95 中位汇总是不同统计量，不能用中位汇总替代业务尾延迟保证。

[集中阶段](RELEASE_VALIDATION.md#已完成的集中结果)中 `split` 相对原算子的 sparse8192 +4% 是另一项对照结论，继续保留。本轮 SDK 对比已优化独立主机的结果不抵销它。CPU 主基准仍是集中阶段的原版原生 BM25S；本轮未重复 CPU，不拼接旧 CPU 数据宣布新的配对加速比。

性能仅覆盖固定 A/B 各八条查询、K10、设备 5、CPU 绑定 0–31/内存节点 0。未验证独占核预留，未扩展为任意查询、其他 K/并发或业务 SLO 保证。完整质量包的少量校验循环不作为吞吐基准。

## 完整查询质量

| 数据集 | 查询数 | nDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| SciFact | 300 | 0.662527869 | 0.779944444 | 0.631021164 |
| TREC-COVID | 50 | 0.576951307 | 0.015566686 | 0.789833333 |
| MS MARCO | 43 | 0.439282594 | 0.130349372 | 0.745994832 |

三组指标与冻结的 NPU `split` 完全一致。与原版 BM25S 的原生同分策略差异仍按集中阶段记录保留，不宣称所有原生 CPU 排名位相同。

## 实际交付位置与身份

以下均为 A5 上已验收的目录，已有数据、旧包和参考目录继续保留：

| 内容 | 路径 |
|---|---|
| 合入后的 SDK 源码 | `/test/w50066029/bm25-npu-indexsdk/results/indexsdk_integrated_01/sdk` |
| SDK 源码评审补丁 | `/test/w50066029/bm25-npu-indexsdk/results/indexsdk_integrated_01/indexsdk-bm25.patch` |
| 实际 SDK 动态库与接口检查程序 | `/test/w50066029/bm25-npu-indexsdk/results/indexsdk_merged_build_01/host` |
| 搬迁消费通过的安装目录 | `/test/w50066029/bm25-npu-indexsdk/results/indexsdk_merged_build_01/relocated` |
| 配套 fused/Merge 别名 | `/test/w50066029/bm25-npu-indexsdk/results/indexsdk_merged_build_01/runtime/vendors` |
| 本轮主机验收与恢复日志 | `/test/w50066029/bm25-npu-indexsdk/results/indexsdk_merged_build_01` |
| 本轮设备报告与所有性能结果 | `/test/w50066029/bm25-npu-indexsdk/results/indexsdk_postmerge_a5_01` |

`libascendfaiss.so` 的 SHA-256 为 `c9052ddbf7bb537b58c99a8506309a66b3c05b53ddf9312356fc21f0f5c2113a`。正常算子内核源码仍为 `d7bd438887798cd693c90599866dac45891b86e03ff6c99907358feedcc45948`，设备 `.o` 仍为 `1004d6ba4873b5da8089d238e32765705de9d21abe7bc0f0eafb68014ae01777`。SDK 设备报告 SHA-256 为 `a82a35af20a91802df39cdb1e3e1634246ed5a201b16ac60afab697b7174b6b1`。

新包名为 `bm25_indexsdk_split_1024`；相对集中阶段包，vendor 名相关的环境/卸载脚本变化，设备内核及其余对应安装文件的哈希保持。完整 19 个 fused 和 18 个 Merge 安装文件指纹见验收证据。不能仅凭两版都相同的 `libcust_opapi.so` 判定设备内核身份。

原参考 IndexSDK 目录未覆盖，本次在独立源码副本内完成合入和测试，尚未向上游 IndexSDK 仓库提交。源码和复现入口随本仓库 `main` 同步；设备二进制、原始数据及报告保留在上述 A5 目录，`git pull` 不会下载这些产物。后续有实现改动再按影响范围回归，本次无需继续调参或重复测量。
