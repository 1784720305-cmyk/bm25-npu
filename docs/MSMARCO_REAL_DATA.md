# MS MARCO 884 万篇真实语料测试

使用完整 **8,841,823** 篇 MS MARCO Passage 语料，保持原版 BM25S、Q8/K10、六轮正反顺序对照以及现有生产结构、安全编译参数。全量输入准备和原版 BM25S 建库已在 A5 通过，首轮 NPU 质量运行遇到内存分配失败，**性能尚未开始**。主机分配策略修复和后续状态见下文。

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

词表确实超过原文件构建器的 100 万上限。因此只将外部变长查询模式的文件构建上限扩大到 2^24，合成模式仍为 65,536；不删除或合并词项，INT32 posting/Q*N、磁盘和实际设备内存检查继续生效。生产 C++ 与算子原本支持这些 INT32 词 ID，目录索引乘法为 uint64，不修改索引或评分合同。首轮失败后调整的是主机内存分配策略。

质量和性能各保留独立索引/oracle，准备和运行产物的额外空间估算约 32.58 GiB，另有原始 ZIP、编码语料和映射文件，以及程序的 20% 磁盘余量检查；开始前建议至少留 **60 GiB** 空闲磁盘。`resource_plan.json` 保存实际 posting 数推导的预算。该估算覆盖已知缓冲与文件，不保证 Python/Numba/运行库的完整峰值。

用户已回传既有设备探测报告：free=131728125952、total=132338679808（约 122.7 GiB 空闲），`/test` 可用 890G。这是总空闲量，不能保证指定页策略下的单块内存可分配。正式基准会在新输出目录重新探测显存，并检查实际 workspace 分配；总量预检查通过也不能替代 `aclrtMalloc` 实际结果。

## A5 执行

首轮 `266e938` 复用了已验收二进制。以下为首次数据准备的历史命令；当前已准备好的 `msmarco_data_01` 不重建，按下一节在新目录编译修复后的主机程序。只复制命令框内容；若终端停在 `>`，先按 Ctrl+C 退出续行。已有数据、二进制、结果和 vendor 均保留。

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
python3 -u standalone/beir_real.py --dataset msmarco prepare --output results/msmarco_data_01
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

该缩小语料仅验证工具与合同，不能作为 884 万篇质量、A5 或 NPU 性能证据。全量成绩仍待 A5 修复后的运行归档。

## 分配失败后的复测

2026-11-05 回传结果确认全量准备成功，`N=8841823`、`V=1624803`、`M=264520096` 均与本地扫描相同，质量/性能索引各为 2171025704 字节。原版 BM25S 公开建库和完整分数参考检查通过后，NPU 程序在 `aclrtMalloc` 返回 207001 / `Memory_Allocation_Failure(EL0004)`；没有六轮性能或通过的 NPU 质量结果。

旧分配器固定使用 `ACL_MEM_MALLOC_NORMAL_ONLY`。目录是第一个主机显式申请的索引缓冲，负载 14051296344 字节、加两侧 guard 后请求 **14051296472 字节**，约 13.1 GiB。旧日志没有记录失败请求大小，暂不能仅凭该日志断言究竟是哪次申请、哪种页资源限制触发失败。

修复将主机 `Buffers::Allocate` 改为 `ACL_MEM_MALLOC_HUGE_FIRST`，与固定参考 IndexSDK `59213c4` 的 `MemorySpace::DEVICE_HUGEPAGE` 一致：优先大页，可按运行库规则回退普通页。[官方内存策略说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/appdevgapi/aclpythondevg_01_0962.html)。失败信息新增真实申请字节数和策略名，仍保留 ACL 错误码、原始错误及全部 guard/生命周期检查。不是修改算子或缩减语料；是否解决本机问题，仍以同尺寸探测和真实运行结果为准。

本地新建目录 `results/indexsdk_allocator_cpu_01` 编译及 CTest 7/7 通过；40 项 Python 工具测试通过。C++ mock 测试检查了超过 4 GiB 的大小不被截断、策略、207001 原始错误、零字节处理和准确释放；Python mock 检查了两种策略分别失败/成功及异常后的释放。它们不证明 A5 真正分配成功。[本次修复记录](../evidence/msmarco_allocation_fix_20261105.json)与历史通过记录分别保留。

先更新并做实际分配探测。每次只申请一个同尺寸目录块，随后释放，不上传索引、不运行算子或 Search。工具只在 `HUGE_FIRST` 成功并完成清理后返回成功；普通页失败是允许记录的诊断结果。

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 -u tools/probe_device_allocation.py --input results/msmarco_data_01/quality --device 5 --output results/msmarco_allocation_probe_01.json
```

如果两种策略都失败，回传探测 JSON，保留现场。若普通页失败而大页优先成功，这支持分配策略解释；即使两者都成功，也需运行新主机检查确认完整分配序列。

探测成功后，在**新输出目录**做真实 CANN 主机构建、单测、生产 clang-tidy、安装消费检查，复用原有 SDK 主机库和 vendor，不覆盖历史构建：

```bash
python3 tools/run_a5_acceptance.py --phase host --output results/indexsdk_alloc_host_01 --toolkit-home /usr/local/Ascend/cann-9.2.0/x86_64-linux --indexsdk-source /test/w50066029/IndexSDK-reference --sdk-library /test/w50066029/bm25-npu-indexsdk/results/indexsdk_host_59213c4_01/libascendfaiss.so --vendors /test/w50066029/bm25-npu/results/final_runtime/vendors --securec-include /usr/local/Ascend/cann-9.2.0/x86_64-linux/include --clang-tidy /test/w50066029/bm25-tools/bin/clang-tidy --jobs 8
```

host 阶段全部通过后，先复用原千万篇合成输入完成选择、合并、共享 SDK 生命周期及完整 Search 回归。这个验收的 C++ CPU 门槛是历史内部检查；MS MARCO 业务性能基准仍为下方的原版 BM25S。

```bash
export MX_INDEX_USE_ONLINEOP=1
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
numactl --physcpubind=0-31 --membind=0 python3 tools/run_a5_acceptance.py --phase device --output results/indexsdk_alloc_host_01 --device 5 --input /test/w50066029/bm25-npu/results/large_scale_close1/inputs/10000000
```

然后直接复用已完成的数据准备，换新的结果目录，使用刚编译的主机程序运行 MS MARCO：

```bash
numactl --physcpubind=0-31 --membind=0 python3 -u standalone/beir_real.py --dataset msmarco run --data results/msmarco_data_01 --output results/msmarco_a5_02 --vendors /test/w50066029/bm25-npu/results/final_runtime/vendors --host-build results/indexsdk_alloc_host_01/build --dependency-dir /test/w50066029/bm25-npu/results/bm25s_numba_py39 --device 5
```

准备输入无需重建；新运行会在计时外重新构造并校验原版 BM25S 对照索引。上述任一步失败，保留新输出，回传该阶段的日志；NPU 错误直接查看 `results/msmarco_a5_02/quality/round1/npu/run.log`。大页策略可能影响内存访问性能，新的性能成绩须单独记录，不能直接套用此前约 2 ms 的成绩。
