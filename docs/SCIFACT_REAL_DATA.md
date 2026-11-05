# SciFact 公开真实数据测试

用户暂无真实数据，选择优先使用容易复现的公开数据集。本轮采用 BEIR SciFact，原始来源为 [BEIR 数据集列表与下载示例](https://github.com/beir-cellar/beir#beers-available-datasets)，数据说明为 [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) 和 [测试 qrels](https://huggingface.co/datasets/BeIR/scifact-qrels)。这是 5183 篇真实科学文献、300 条有标注的测试查询及 339 条正相关判断的小型检索测试，不是千万篇规模测试。

## 数据及验证口径

官方 `scifact.zip` 为 2816079 字节，SHA256 固定为 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`。脚本仅读取指定 ZIP 成员，验证完整性、数量、唯一 ID 和 qrels 引用；原始压缩包与生成的数据保存在 `results`，不提交到代码仓库。

预处理拼接标题与正文，使用与固定原版 BM25S basic tokenizer 一致的小写、Unicode 两字符以上单词、原版英文停用词、无词干化配置。两端共享同一词表、词 ID、文档 ID 映射及 ATIRE/Lucene-IDF、k1=1.2、b=0.75 参数。原始文本和测试 ID 可追溯；查询不截成三词，不去重，只按固定文档词表排除 OOV 并记录数量。本地实际词表为 35867，处理后查询长 3～22，未触及入口的 64 词上限。

评估分成两部分：

1. **全量质量/正确性**：全部 300 条测试查询分成两组各 150 条，执行一次检查轮。NPU 使用独立原文 TF/DF 的 FP32 分数、严格 Top-K、全部叶块/合并层及重复输出检查；原版 BM25S 分别用 1/8/32 线程校验。该阶段导出全部 Top-10，计算 nDCG@10、Recall@10、MRR@10。Q150 只用于全量质量检查，不计为 Q8 性能结果。
2. **Q8 性能**：固定种子 20261105 从这 300 条查询抽取 16 条，不参考相关性标签，构成 A/B 两个 Q8 包。自然词长保持原样，按两包实际最大总词数声明容量。六轮正反顺序测 NPU 和原版 BM25S 1/8/32 线程，5 次预热、每集 30 个样本。建库、分词、JIT 与诊断在计时外，输出完整 Search 的中位批延迟、P95 和 Search 时间 QPS。

原生 BM25S 的同分 ID 策略与 NPU 严格合同可能不同，质量指标分别报告，不把原生耗时胜出冒充严格合同等价。此配置也不冒充采用其他分词/评分参数的公开排行榜成绩。若小库 NPU 较慢，同样如实记录。

## 本地已完成的检查

[本地记录](../evidence/scifact_local_validation_20261105.json)包含：官方包指纹、全部 6292 条原始文档/查询文本与实际固定原版 tokenizer 的一致性、C++ CPU 对全部 300 条质量查询与 Q8 样本的完整分数/严格 ID 位检查、实际原版 NumPy/Numba CPU 建库及原生并行检索。

本机 C++ CPU 与原版 BM25S CPU 的全量 nDCG@10 均为 0.662528、Recall@10 均为 0.779944、MRR@10 均为 0.631021。这些是 Windows CPU 验证，后续 A5 实测结果见下一节。生产 C++、算子和安全编译参数未改动。

## 2026-11-05 A5 真实数据结果

用户已完成 `scifact_data_01` 准备及 `scifact_a5_01` 的全量质量与六轮 Q8 性能测量。[回传记录](../evidence/scifact_a5_20261105.json)保存终端数据和本地重算；原始排名、逐样本数据及各线程配置报告仍在 A5，未单独下载。

全部 300 条查询上，NPU 与原版 BM25S 1/8/32 线程的三项质量指标完全一致：nDCG@10 = **0.662527869**、Recall@10 = **0.779944444**、MRR@10 = **0.631021164**。这说明本轮聚合质量指标对齐，不宣称所有单条排名或同分 ID 位相同；NPU 独立分数/严格 ID 等检查的完成由测量驱动成功继续到最终输出体现。

性能使用固定种子的 16 条查询，分成两个 Q8 包；以下为六轮中位批延迟的中位数：

| 查询包 | 原版 BM25S 同轮较快配置 ms | NPU ms | NPU 延迟相对 CPU | NPU 中位延迟胜出轮数 |
|---|---:|---:|---:|---:|
| A | 0.08346025 | 0.08627200 | 慢 3.369%，多 2.812 微秒 | 0/6 |
| B | 0.07775450 | 0.07358025 | 快 5.368%，少 4.174 微秒 | 6/6 |

A 组中位延迟稳定落后，本轮差距应如实记录，不能笼统用系统负载解释。两组的合并“均六轮领先”门槛未通过：`timing_gate_passed` A 为 false、B 为 true。该字段判定的是相对 CPU 的性能，false 不表示质量/正确性检查失败。程序最终打印 `REAL DATA QUALITY`/`REAL DATA PERFORMANCE` 代表检查与测量完成，也不代表所有性能门槛通过。

P95 的六轮汇总为 A：CPU 0.0896895 ms、NPU 0.0879325 ms，NPU 4/6 轮较低；B：CPU 0.0845665 ms、NPU 0.0757240 ms，NPU 6/6 轮较低。当前摘要没有逐样本或均值数据，不从中位数倒推实测总体 QPS。

本轮比较 NPU 与原版 BM25S，不是整改前后 NPU 二进制的配对；不把 A 组的相对差距归为结构或安全参数导致的回退。固定输入的本地复现中，A 包共有 88 个词、B 包 67 个词，命中分布也不同，但 A5 输入文件指纹和逐线程报告未回传，因此这些统计仅作输入核查参考，不能直接用来证明某项性能原因。

结论为：**真实数据质量通过，小库性能有胜有负，整体不宣称领先原版 BM25S**。质量覆盖全部 300 条测试查询，性能只覆盖固定 16 条；更广泛的查询分布及更大真实语料需要各自的测量。保留现有生产结构和安全参数，不为本轮几微秒差异直接修改生产策略。

## A5 执行

上述首轮已完成；以下为复现方法，已有结果保留，重测时另用新目录。

继续使用已验收主机二进制、vendor 和原私有 Numba 依赖。准备数据可重复使用已核验本地 ZIP；没有指定 `--archive` 时，从上述官方 URL 下载约 2.8 MB。沿用 CANN 环境，并在实际预留核的窗口执行：

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
python3 standalone/scifact.py prepare --output results/scifact_data_01
numactl --physcpubind=0-31 --membind=0 python3 -u standalone/scifact.py run --data results/scifact_data_01 --output results/scifact_a5_01 --vendors /test/w50066029/bm25-npu/results/final_runtime/vendors --host-build results/indexsdk_a5_acceptance_02/build --dependency-dir /test/w50066029/bm25-npu/results/bm25s_numba_py39 --device 5
```

若 A5 无法下载，可把已核验的 `scifact.zip` 复制到 A5，准备命令追加 `--archive /实际路径/scifact.zip`；该参数只读原 ZIP。已有数据/结果目录不覆盖，失败后保留日志并为重试使用新目录。

回传最后的 `REAL DATA QUALITY`、`REAL DATA PERFORMANCE` 和 `REPORT`。分阶段日志位于新结果目录的 `quality.log` 和 `performance.log`。程序完成表示检查与测量完成；实际优劣按本轮结果判读，不沿用合成数据 15.8 倍的结论。
