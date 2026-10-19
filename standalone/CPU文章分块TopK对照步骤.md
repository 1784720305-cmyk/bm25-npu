# 补强CPU单查询Top-K对照

最新实机状态：黄区GNU10.3.1/OpenMP4.5编译成功；1647组C++全排序/原FP32字节检查通过，每组3次覆盖/变换编号位置。原小样本、64K单查询、256K单查询及256K/8查询的完整计算和CPU/NPU直接核对均通过。单查询S=8在初筛中优于原CPU；8查询两种CPU方式各有波动，后续两种CPU与NPU各做六轮对照。更快CPU文章分块路径与NPU基本持平，不能沿用较慢CPU成绩宣称加速。下一步见 [零分汇总实验](Radix零分汇总实验步骤.md)。以下编译和验证命令保留供复现，已存在的输出目录不要重复执行或删除。

## 为什么现在做这个

Radix四段采集已经在设备2完成，68条任务与两张设备表交叉核对通过，排除首次和预热后正式10轮：

| 阶段 | 中位数，微秒 | p95，微秒 |
|---|---:|---:|
| 贡献 | 2.250 | 2.652 |
| 文章总分累加 | 5.380 | 5.669 |
| Radix局部Top-K | 22.721 | 22.767 |
| 最终合并 | 4.232 | 4.366 |

逐轮四段设备耗时之和中位数34.606微秒，整链首尾跨度中位数38.308微秒，链内间隔之和中位数3.702微秒。这些是采集诊断，不是无采集基准，不把各段中位数直接相加当成整批中位数。

Top-K仍是最大一段。历史Warp局部Top-K为40.643微秒，本次Radix为22.721微秒，支持继续保留Radix；两次采集并非同期配对实验，不据此宣称固定算法加速比。单靠总耗时也不能判断是原子计数、重复扫描还是同步占主导。

另一方面，当前CPU的贡献和累加可多线程，Top-K却按查询并行。只有一条查询时，Top-K扫描文章主要由一个线程完成；“8线程设置”不表示这段也分给8个线程。为让性能对手更合理，先加入可选的文章分块CPU对照，暂不继续修改NPU内核。

## 输入、步骤、输出及验收

输入沿用原TF、长度项、IDF和分组数组，先按原公式得到完整文章分数。贡献公式和累加顺序不改。

新CPU Top-K可将一条查询的文章分成S份。例如65536篇、8份，每份8192篇，由多个CPU线程扫描；每份保留至多K个候选，再从所有候选里选全局K个，只在最后排序K项。局部不要求排序，减少不必要的局部排序工作。分块可以不等长，分块内文章少于K时保留全部，空候选不参与合并。

不会漏掉全局前K：如果一篇文章在自己分块中已经排不到前K，那么仅这个分块就有至少K篇文章按统一规则排在它前面，它也不可能进入全局前K。所有步骤保持分数降序、同分原文章编号升序；保留原分数字节，正负零视为数值同分。

这是本项目原有有界堆与已使用的两级分块选择思路，不新增算法库。组内IndexSDK的TopkOp也有CPU堆筛选、按batch并行代码，但不能把本次单查询文章分块称作直接复制其现成实现。

输出仍是贡献、文章总分、最终Top-K分数与编号四数组。验收分三步：C++独立选择单元检查、原小样本公式核对、正常65536文章与既有Radix结果交叉。通过后才测性能；新版本未必更快，多一次分块合并和线程调度也有成本。

## 本次代码内容和状态

- `cpu_reference/topk_document_shards.h`：`KeepCandidate`复用有界堆规则；`TopKDocumentShards`先并行处理查询×文章分块任务，写出无序局部候选，再合并每条查询的候选，最后排序K项。临时缓冲由主程序在计时前分配，各轮重写。
- `cpu_reference/cpu_runner.cpp`：原贡献、累加、原Top-K函数体保持原文本；新增可选末尾参数TOPK_DOCUMENT_SHARDS。0沿用原按查询并行路径，正数启用文章分块，须不超过线程数及文章数。报告明确记录模式，不自动替换默认路径。
- `cpu_reference/topk_document_shards_test.cpp`：用独立的完整数值排序核对所有编号及原FP32字节，覆盖全零、全等、正负零、次正规数、最大有限数、随机同分、K=1/10/32、小分块和不等长分块；复用内存并变换编号位置，输出和候选先填无效值。该C++检查现已由用户在黄区实际通过。
- `standalone/run_cpu.py --topk-document-shards S`：传递明确模式，拒绝旧二进制没有报告新模式的情况。
- `benchmark_case.read_cpu_result`：性能读表时核对请求的CPU模式，避免把原CPU时间和新CPU时间混在一起。
- `confirm_sharded.py --cpu-topk-document-shards S`：后续可用指定CPU模式做CPU/NPU轮换复测，报告记录模式；原默认0不变。

本机374项Python模型/调用保护检查通过，新CPU模式6轮模拟复测及原默认/轮换/数据工具回归通过。它们不等于C++已编译、CPU实机已正确或速度已提高。NPU源码、调用器和二进制样本未改，无需重新编译或运行NPU。

## 黄区先编译CPU和执行C++单元检查

提交、同步本次代码后逐行执行，任何一步失败就停止。本次使用独立CPU构建目录，保留原CPU二进制作对照：

```bash
cd /test/w50066029/bm25-npu
python3 verify_package.py
set -o pipefail
cmake -S cpu_reference -B results/cpu_document_host_build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON 2>&1 | tee results/cpu_document_configure1.log
cmake --build results/cpu_document_host_build -j 2>&1 | tee results/cpu_document_build1.log
```

编译成功后执行单元检查；使用直接可执行文件，不依赖ctest命令版本：

```bash
results/cpu_document_host_build/bm25_cpu_topk_tests 2>&1 | tee results/cpu_document_topk_tests1.log
```

预期是完整编号及原分数字节核对通过，并输出实际通过组数。这只执行CPU，不使用NPU。

## 单元检查通过后，先原6篇文章

使用4线程、2个文章分块，保留原输入：

```bash
python3 standalone/run_cpu.py --input results/pipeline_input --output results/cpu_document_small_s2_check1 --runner results/cpu_document_host_build/bm25_cpu_runner --threads 4 --warmup 2 --repeats 3 --topk-document-shards 2 2>&1 | tee results/cpu_document_small_s2_check1.log
python3 standalone/pipeline_fixture.py check --input results/pipeline_input --actual results/cpu_document_small_s2_check1 2>&1 | tee results/cpu_document_small_s2_check1_validation.log
```

## 小样本通过后，正常65536文章

```bash
python3 standalone/run_cpu.py --input results/normal_64k_q1 --output results/cpu_document_64k_q1_s8_check1 --runner results/cpu_document_host_build/bm25_cpu_runner --threads 8 --warmup 2 --repeats 3 --topk-document-shards 8 2>&1 | tee results/cpu_document_64k_q1_s8_check1.log
python3 standalone/compare_compute_outputs.py --input results/normal_64k_q1 --cpu results/cpu_document_64k_q1_s8_check1 --npu results/radix_sharded_64k_q1_s32_check1 --report results/cpu_document_64k_q1_s8_cross1.json 2>&1 | tee results/cpu_document_64k_q1_s8_cross1.log
```

这里NPU只读取既有实际输出文件，不重新执行。新CPU输出必须符合独立公式并与同一输入Radix结果一致。不放宽同分规则，也不拿CPU答案替代NPU。

## 正确后如何决定性能对手

先使用这个新CPU二进制分别测原路径S=0和文章分块路径，保持输入、线程及预热/测量数一致；再检查合理线程/分块配置，选实测更快者作为CPU对手。不要因为某个CPU实现更慢就选它来放大NPU收益。

可选模式已接入6轮CPU/NPU轮换入口，该模式已在黄区用于正确性通过后的同期复测，具体最新结论见页首。即使S=8正确，也不代表S=8最优或整个服务器CPU已优化到上限。后续仍需同期旧版/Warp/Radix对照、原版BM25S、规模/批量曲线和含传输/完整查询验收。

建议commit备注：`feat: 增加CPU文章分块Top-K可选对照及正确性检查`
