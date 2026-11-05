# TREC-COVID 中等规模真实数据测试

沿用用户认可的 Q8、原版 BM25S 和现有安全参数，在 SciFact 5183 篇之后测试完整的 BEIR TREC-COVID：**171332 篇文档、50 条测试查询**，文档数约为 SciFact 的 33 倍。[官方数据说明](https://huggingface.co/datasets/BeIR/trec-covid)、[BEIR 下载与评估入口](https://github.com/beir-cellar/beir#beers-available-datasets)。

这是新的真实工作负载，不是简单复制 SciFact 扩库，也不是千万篇业务语料。两份数据的词频和查询分布不同，不把两个结果的变化归为仅由文档数造成。首轮 A5 已完成，两组均稳定领先，范围与数值见下文。

## 数据与共用入口

使用官方 `trec-covid.zip`，73876720 字节，约 74 MB；SHA256 为 `120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0`。脚本固定大小、指纹和数据数量，失败时保留现场，不换源或缩减语料。

共用入口为 `standalone/beir_real.py`，旧 `standalone/scifact.py` 命令仍可使用。SciFact 的已有数据、结果和指标保持原样。较大语料采用 ZIP 文本流的多次遍历，词表和文档 ID 映射保留在内存，正文及全部分词结果不常驻 Python 列表。TREC-COVID 保留官方文档顺序，SciFact 保留此前的文档 ID 排序。

本地实测词表 **215004**，词 ID 超过 65535。仅外部变长查询的文件构建器上限扩大至 100 万词，原合成模式仍限制 65536；字节、INT32、posting 数和磁盘预算仍检查。C++ 接口原本使用 INT32，生产 C++、算子和编译策略均未修改。文档与查询不复制扩容、截断或哈希合并词；两端使用相同词表和同一原版 basic tokenizer 配置。

后续 [MS MARCO 接入](MSMARCO_REAL_DATA.md)将外部模式的词表构建上限进一步扩大到 2^24，以保留完整的 162 万词表；本次 TREC-COVID 的索引与历史成绩保持原样。

## 质量与性能

- **质量**：50 条测试查询全部评估，分成两个 Q25 包，检查完整分数、严格 Top-K、叶块/合并层、索引及重复输出。原版 BM25S 分别测 1/8/32 线程；导出全部 Top-10，报告 nDCG@10、Recall@10、MRR@10。
- **性能**：固定种子抽取 16 条测试查询，分成两个 Q8 包，六轮正反顺序，每集 5 个预热周期、30 个样本；原版 BM25S 取同轮 1/8/32 线程较快者。Q25 质量阶段不计作 Q8 性能。建库、分词、JIT、首次上传和诊断均在 Search 计时外。
- 保留原生同分排序，不把严格适配器放入 CPU 性能基准；原生与项目严格合同的排序/质量差异分开记录，不强迫指标相同。

这份 qrels 有 66336 行：41661 个 0、10456 个 1、14217 个 2，以及 2 个 -1（未判定）。nDCG 使用 trec_eval 的线性 relevance gain；Recall/MRR 仅把 relevance > 0 当作相关，0/-1 不增加 gain。依据见 [trec_eval 定义与实现](https://raw.githubusercontent.com/usnistgov/trec_eval/master/m_ndcg_cut.c)。SciFact 只有二元正标注，该共用实现保持其原数值。

TREC-COVID 一条查询有较多相关文档，Recall@10 的分母与 SciFact 不同；指标用于同一语料、同一 K 下比较 NPU 与原版 CPU，不直接比较两个数据集的 Recall 高低。

## 本地验证范围

[本地验证记录](../evidence/trec_covid_local_validation_20261105.json)保存了 33 项工具测试、完整数据集的 C++ CPU 分数/严格 ID 位验证、原版 NumPy/Numba 的建库/公式/原生检索检查，以及全部 171382 条原文与查询的实际 tokenizer 一致性检查。SciFact 共用入口复现了旧版的相同索引、查询包及 oracle 字节。

本机 C++ 严格排序的 nDCG@10 为 0.576951，原生 BM25S CPU 为 0.574552；原版 full-score 严格 ID 与项目一致，原生 retrieve 在两组分别有 46/37 个位置的同分 ID 差异，已保留原始差别。此处为 Windows CPU 检查，不能作为 NPU 成绩或 A5 速度。

## A5 执行

首轮 `trec_covid_a5_01` 已完成，无需为归档重复运行；以下用于复现，已有目录保留，重测时换用新目录。

当前额外复测使用已完成 A5 验收的大页分配版本 `results/indexsdk_alloc_host_01/build`；本页历史成绩保持原样，不预设新会话速度。已准备好的输入可继续复用。

在相同资源条件下继续使用已验收二进制、vendor 和私有 Numba。数据、结果使用新目录：

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
python3 standalone/beir_real.py --dataset trec-covid prepare --output results/trec_covid_data_01
numactl --physcpubind=0-31 --membind=0 python3 -u standalone/beir_real.py --dataset trec-covid run --data results/trec_covid_data_01 --output results/trec_covid_a5_02 --vendors /test/w50066029/bm25-npu/results/final_runtime/vendors --host-build results/indexsdk_alloc_host_01/build --dependency-dir /test/w50066029/bm25-npu/results/bm25s_numba_py39 --device 5
```

准备阶段需要下载并多次遍历原始文本和构建独立 oracle，该耗时与检索耗时分开。若采用已下载的官方 ZIP，准备命令可追加 `--archive /实际路径/trec-covid.zip`；大小和 SHA256 必须一致。压缩包及生成文件保存在 `results`，不提交到代码仓库。

回传 `REAL DATA QUALITY`、`REAL DATA PERFORMANCE`、`REPORT`。阶段日志分别为 `quality.log`、`performance.log`。完成检查不自动表示性能领先，需查看各组判定。

## 2026-11-05 A5 结果

用户回传了完整质量汇总和六轮性能判定，归档于 [本轮证据](../evidence/trec_covid_a5_20261105.json)。本地重算倍率、延迟降幅及六轮中位/P95 胜出次数；原始报告、逐样本数据、排名和每线程配置报告仍在 A5，未单独下载。

每批 Q8 的六轮中位延迟汇总为：

| 查询包 | 原版 BM25S 同轮较快配置 ms | NPU ms | CPU/NPU 延迟比 | NPU 耗时降低 | 中位/P95 胜出轮数 |
|---|---:|---:|---:|---:|---:|
| A | 0.34925650 | 0.24825475 | 1.40685 | 28.92% | 6/6、6/6 |
| B | 0.30292425 | 0.21513250 | 1.40808 | 28.98% | 6/6、6/6 |

P95 的六轮中位汇总为 A：CPU 0.380926 ms、NPU 0.250806 ms；B：CPU 0.358868 ms、NPU 0.2177815 ms。两组 `timing_gate_passed` 均为 true。约 1.41 是延迟比，不等于耗时减少 41%；本轮耗时减少约 29%。摘要未提供均值或原始样本，不由中位数倒推实测总体 QPS。

全部 50 条测试查询的质量为：

| 实现 | nDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| NPU | 0.576951307 | 0.015566686 | 0.789833333 |
| 原版原生 BM25S（1/8/32 线程一致） | 0.574552146 | 0.015476670 | 0.789833333 |

两端指标分别精确复现了此前本地严格 C++ 参考和原版原生 CPU 的指标，与已记录的同分 ID 差异一致。此前本地原版 full-score 严格 ID 与项目一致、原生选择分数集合相同，只在同分 ID 上有区别。该证据支持同分策略解释，不据此宣称 NPU 提升了检索质量；本轮 A5 单条排名没有单独回传。`strict_equivalent_win: false` 保留原生排序合同差别，不表示 NPU 的独立正确性检查失败。

本轮结论：**17.1 万篇真实语料上，两组 Q8 样本六轮均有约 1.41 的延迟比，独立检查完成，生产结构与安全参数保留**。质量覆盖全部 50 条查询，性能覆盖固定 16 条抽样查询。它不扩展为全部查询、百万/千万真实语料或服务吞吐的通用结论，也不把与 SciFact 的差别单独归因于文档数。

## 大页分配修复后的补测

`trec_covid_a5_02` 已使用 `results/indexsdk_alloc_host_01/build` 完成质量与固定 Q8 六轮性能复测。NPU 和原版 1/8/32 线程各自的全部质量指标均与本页历史值精确相同，原有并列排序差异继续保留。[补测证据](../evidence/small_medium_regression_a5_20261105.json)保存原值和独立重算。

| 查询包 | 旧 NPU 中位 ms | 新 NPU 中位 ms | NPU 变化 | 本轮原版 CPU ms | CPU/NPU | 中位/P95 胜出轮数 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.24825475 | 0.24873300 | +0.19% | 0.35140700 | 1.41279 | 6/6、6/6 |
| B | 0.21513250 | 0.21408375 | -0.49% | 0.30256350 | 1.41330 | 6/6、6/6 |

NPU 中位数相对旧值变化 **+0.47825/-1.04875 微秒**。P95 的六轮中位汇总从 0.250806/0.2177815 ms 变为 **0.251016/0.2169105 ms**，变化 **+0.08%/-0.40%**。两组 `timing_gate_passed` 均为 true，NPU 相对本轮原版 CPU 耗时分别低 **29.22%/29.24%**。

该跨会话结果显示当前中等规模工作负载的 NPU 延迟接近历史值，不把微小差异归为页策略的独立效果。新单条排名、逐样本和运行指纹未单独回传，已明确记录证据边界。补测已完成，无需重复运行；历史结果和原数据保留。
