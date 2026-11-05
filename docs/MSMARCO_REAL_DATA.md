# MS MARCO 884 万篇真实语料测试

使用完整 **8,841,823** 篇 MS MARCO Passage 语料，保持原版 BM25S、Q8/K10、六轮正反顺序对照以及现有生产结构、安全编译参数。此入口已完成本地校验，**全量 A5 性能尚未执行**。

## 固定数据来源与评测范围

官方 [BEIR 下载表](https://github.com/beir-cellar/beir#beers-available-datasets)提供 `msmarco.zip`；[原始数据说明](https://huggingface.co/datasets/BeIR/msmarco)确认 8,841,823 篇。压缩包实际校验如下：

- 下载量：1,082,258,632 字节，约 1.08 GB。
- 官方表中 MD5：`444067daf65d982533ea17ebd59501e4`，本地下载文件完全匹配。
- 本地固定 SHA256：`f46aed8343564261c27228b7cf22dde5c9a70143fe34dc3c589bd5322308712a`。
- 全部查询文件有 509,962 条查询；本轮只使用 `qrels/test.tsv` 对应的 **43 条查询、9,260 条分级标注**，等级为 0/1/2/3。**6,980 条是 `dev` 查询，本轮不宣称覆盖 dev**。[test 分割数量说明](https://huggingface.co/datasets/irds/beir_msmarco_test)。

质量阶段完整评估这 43 条 test 查询。两个包各 Q22，B 包最后一个 `null` 查询 ID 对应空查询，仅满足相同容量要求；其分数和运行仍受检查，排名不进入质量指标。性能阶段独立从 43 条查询按固定种子 20261105 抽取 16 条，两个 Q8 包，维持自然词序、长度及重复词，不截断、去重或复制扩库。

分词与之前保持一致：原版 basic tokenizer、小写、原版英语停用词、无词干化；BM25 为 atire TF、lucene IDF、k1=1.2、b=0.75、FP32。指标沿用已声明的 BEIR 比较口径：nDCG 使用线性 relevance gain，Recall/MRR 把 relevance > 0 计为相关；不与采用 relevance >= 2 等其他阈值的 TREC 结果混报，也不作为官方排行榜复现成绩。

原版 `.retrieve` 的同分行为保留，测 1/8/32 线程，逐轮采用较快配置；完整分数与项目独立原文 oracle 对照，同分 ID 差异单独记录。分词、建库、JIT、首次索引上传、文件 IO 和诊断不计入 Search。

## 实际语料扫描与资源

本地已遍历整个官方语料，在上述分词配置下得到：

| 项目 | 数量 |
|---|---:|
| 文档 | 8,841,823 |
| 词表 | 1,624,803 |
| 唯一 term/document posting | 264,520,096 |
| 保留词元 | 340,859,891 |
| 叶分片 | 1,080 |
| NPU 常驻索引与目录 | 16,209,345,764 字节，约 15.1 GiB |
| 含质量诊断与运行器余量的空闲显存门槛 | 约 21.03 GiB |

词表确实超过原文件构建器的 100 万上限。因此只将外部变长查询模式的文件构建上限扩大到 2^24，合成模式仍为 65,536；不删除或合并词项，INT32 posting/Q*N、磁盘和实际设备内存检查继续生效。生产 C++ 与算子原本支持这些 INT32 词 ID，目录索引乘法为 uint64，均无需修改。

质量和性能各保留独立索引/oracle，准备和运行产物的额外空间估算约 32.58 GiB，另有原始 ZIP、编码语料和映射文件，以及程序的 20% 磁盘余量检查；开始前建议至少留 **60 GiB** 空闲磁盘。`resource_plan.json` 保存实际 posting 数推导的预算。该估算覆盖已知缓冲与文件，不保证 Python/Numba/运行库的完整峰值。

用户已回传现有设备探测报告：free=131728125952、total=132338679808（约 122.7 GiB 空闲），`/test` 可用 890G。本次重复探测提示 `Memory output exists`，表明同名报告已存在；保留文件即可。正式基准会在新输出目录重新探测显存，并检查实际 workspace 分配。

## A5 执行

仅更新 Python 工具和文档，复用已验收主机二进制及 vendor，无需重新编译算子。只复制命令框内容；若终端停在 `>`，先按 Ctrl+C 退出续行。已有 SciFact、TREC-COVID 和合成数据目录保留。

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
python3 -u standalone/beir_real.py --dataset msmarco prepare --output results/msmarco_data_01
numactl --physcpubind=0-31 --membind=0 python3 -u standalone/beir_real.py --dataset msmarco run --data results/msmarco_data_01 --output results/msmarco_a5_01 --vendors /test/w50066029/bm25-npu/results/final_runtime/vendors --host-build results/indexsdk_a5_acceptance_02/build --dependency-dir /test/w50066029/bm25-npu/results/bm25s_numba_py39 --device 5
```

准备步骤下载、遍历原文、生成索引和独立分数参考，耗时不属于检索延迟。运行在 `PREPARED` 后才开始质量与六轮性能阶段，不要把准备耗时当作 Search 速度。

如果已单独下载官方 ZIP，准备命令追加 `--archive /实际路径/msmarco.zip`；仍检查固定大小和 SHA256。准备中断时保留原目录，使用新的数据输出目录；完整的旧 `source.zip` 可经 `--archive` 复用。运行中断后也保留输出，重试使用新的 `--output`，无需重建已完整准备的数据。

回传末尾的 `REAL DATA QUALITY`、`REAL DATA PERFORMANCE`、`REPORT` 即可；错误分别查看新输出内的 `quality.log` / `performance.log`。实际文档和查询分布不同，不预设恢复合成库的 15 倍倍率，也不由中位延迟倒推实测总体 QPS。

## 本地验证边界

[本地验证记录](../evidence/msmarco_local_validation_20261105.json)保存完整语料扫描、源文件校验及资源推导。另从官方全量语料选择全部 test-judged 文档形成 **9139 篇**的本地验证语料，保留全部 43 条查询及 9260 条原始标注，实际执行：

- 两份准备输入、Q22/Q8、一个空补齐行、文件指纹及索引一致性检查。
- 冻结原版 NumPy/Numba 的公开建库、完整分数与严格 ID 对照、原生 1/2 线程检索和重复输出检查。
- 现有 C++ CPU 程序的完整 FP32 分数与严格 Top-K ID 字节一致性检查。
- 9139 篇验证文档及整个语料均匀取样文本、全部 43 条 test 查询的原版 tokenizer 一致性检查。

该缩小语料仅验证工具与合同，不能作为 884 万篇质量、A5 或 NPU 性能证据。全量成绩仍待上述 A5 命令运行后归档。
