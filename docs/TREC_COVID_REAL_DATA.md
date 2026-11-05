# TREC-COVID 中等规模真实数据测试

沿用用户认可的 Q8、原版 BM25S 和现有安全参数，在 SciFact 5183 篇之后测试完整的 BEIR TREC-COVID：**171332 篇文档、50 条测试查询**，文档数约为 SciFact 的 33 倍。[官方数据说明](https://huggingface.co/datasets/BeIR/trec-covid)、[BEIR 下载与评估入口](https://github.com/beir-cellar/beir#beers-available-datasets)。

这是新的真实工作负载，不是简单复制 SciFact 扩库，也不是千万篇业务语料。两份数据的词频和查询分布不同，不把两个结果的变化归为仅由文档数造成。是否领先及倍率以本轮 A5 实测为准。

## 数据与共用入口

使用官方 `trec-covid.zip`，73876720 字节，约 74 MB；SHA256 为 `120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0`。脚本固定大小、指纹和数据数量，失败时保留现场，不换源或缩减语料。

共用入口为 `standalone/beir_real.py`，旧 `standalone/scifact.py` 命令仍可使用。SciFact 的已有数据、结果和指标保持原样。较大语料采用 ZIP 文本流的多次遍历，词表和文档 ID 映射保留在内存，正文及全部分词结果不常驻 Python 列表。TREC-COVID 保留官方文档顺序，SciFact 保留此前的文档 ID 排序。

本地实测词表 **215004**，词 ID 超过 65535。仅外部变长查询的文件构建器上限扩大至 100 万词，原合成模式仍限制 65536；字节、INT32、posting 数和磁盘预算仍检查。C++ 接口原本使用 INT32，生产 C++、算子和编译策略均未修改。文档与查询不复制扩容、截断或哈希合并词；两端使用相同词表和同一原版 basic tokenizer 配置。

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

在相同资源条件下继续使用已验收二进制、vendor 和私有 Numba。数据、结果使用新目录：

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
python3 standalone/beir_real.py --dataset trec-covid prepare --output results/trec_covid_data_01
numactl --physcpubind=0-31 --membind=0 python3 -u standalone/beir_real.py --dataset trec-covid run --data results/trec_covid_data_01 --output results/trec_covid_a5_01 --vendors /test/w50066029/bm25-npu/results/final_runtime/vendors --host-build results/indexsdk_a5_acceptance_02/build --dependency-dir /test/w50066029/bm25-npu/results/bm25s_numba_py39 --device 5
```

准备阶段需要下载并多次遍历原始文本和构建独立 oracle，该耗时与检索耗时分开。若采用已下载的官方 ZIP，准备命令可追加 `--archive /实际路径/trec-covid.zip`；大小和 SHA256 必须一致。压缩包及生成文件保存在 `results`，不提交到代码仓库。

回传 `REAL DATA QUALITY`、`REAL DATA PERFORMANCE`、`REPORT`。阶段日志分别为 `quality.log`、`performance.log`。本轮 A5 测量尚待执行，完成检查不自动表示性能领先。
