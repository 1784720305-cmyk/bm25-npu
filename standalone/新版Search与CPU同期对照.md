# 新版完整Search与优化CPU同期对照

**后续实际结果：该同期对照已在设备4全部通过。** A CPU0.0742165ms/NPU0.332506ms，B CPU0.07401325ms/NPU0.3325485ms，均为六轮中位数再取中位数；NPU均快0/6轮，约慢4.48/4.49倍。正确性已通过、加速目标未达成；下文保留复现，不要求重跑，当前优先执行[融合优化步骤](融合评分TopK优化步骤.md)，独立源码和生成器已完成、待设备编译；下文原操作保留复现，BM25S作为辅助对照。

## 当前结论和本轮目标

已由用户在950设备4完成局部Top-K六轮完整Search对照，评分和合并固定：A原Top-K六轮中位数再取中位数0.38595125ms、压缩版0.3321105ms，下降13.95%；B为0.3821755ms和0.3317205ms，下降13.20%。A/B均压缩版快6/6轮，所有完整分数、局部候选和最终输出的字节检查通过。保留“连续分区评分＋正分压缩Top-K＋现有合并”作为后续候选组合。上述8条查询整批时间不能当单条延迟，也不是超过CPU的证据。

本轮用这套已验证组合，与同输入的优化C++ CPU同期比较。不继续改内核，不重新扫描CPU线程，不重做37组或msprof。原版BM25S Numba同边界对照仍在后续任务中，不用其历史NumPy结果代替本轮CPU对手。

## 输入、处理、输出、验收

输入复用`results/search_256k_q8_input1`：262144篇文章、8条查询、K=10，索引是原始TF倒排数据及FP32 IDF/长度项。两边均提前在各自内存保存索引、按查询计算同一公式，不预计算贡献。A为原查询，B为同长度变化词编号。

CPU采用此前此数据在1/8/16/32/64线程扫描中选出的32线程、32文章分区配置。此次命令显式固定`--cpu-config 32:32`，重新测时间，不复用历史约0.070ms作为新样本。32是软件线程数，不等于占满整机物理核；报告保留实际线程数、进程亲和性及可见配额，不声称这是所有NUMA/绑核配置的机器最优解。若换输入规模，要重新确认配置依据。

处理步骤：

1. 只读复核已通过的完整Search Top-K对照报告，包含新组合安装源码/API/950目标文件、三组评分和37组Top-K证据指纹、五批完整Search输出、各轮计时会话及检查点。不重新执行这些NPU验收或借用其耗时。
2. 新建独立运行目录，让现有NPU程序加载已验证的新组合；旧vendor和旧运行目录保留。复用现有CPU/NPU主机程序及计时检查，不重编译。
3. 在CPU上实际执行A/B/A/全空/B五次Search，验证全部文章分数、完整严格排名、重复字节，并与已有NPU五批输出直接核对；实际线程必须达到指定值。出错停止，不用CPU答案替换NPU。
4. 六轮独立进程交替“CPU/NPU、下一轮NPU/CPU”。每个进程5个预热周期、30个正式周期，每周期A/B也交替；计时前后核对完整分数、严格排名和重复输出，跨CPU/NPU核对四个检查点。CPU分数与有序FP32参考逐字节一致；NPU与参考按原容差检查，最终编号及顺序严格一致，各自选中分数保留原始字节，不强制不同设备计算的完整分数逐字节相等。

两边均测“CPU已有原始分组词编号 → 开始计时 → Search → CPU拿到合规Top-K → 停止”。NPU包含入口验证/打包/查询上传、设备索引检索/计算、同步、结果下载和必要检查/请求资源释放；CPU包含对应入口验证、索引检索、FP32评分和Top-K，没有伪造传输开销。静态索引加载、冷初始化/容量预启动单列；调试下载、输出污染、文件读写、独立公式核对在正式计时外。原C++边界未改。

输出`report.json`包含所有正式样本、A/B各轮中位数/p95/吞吐、逐轮CPU时间÷NPU时间及NPU胜出轮数。比值大于1才表示该轮NPU更快；小于1仍是有效结果，不算程序失败。批量吞吐按完整Search时间统计，不将整批时间除以8当单条查询延迟。两边实现的评分组织及Top-K算法不同，这不是“仅换硬件”的实验。

## 执行

Windows提交，务必包含新文件：

```powershell
cd F:\Ascend\bm25\bm25-a5
git add -- standalone/benchmark_compact_search_cpu.py standalone/test_compact_search_cpu.py package_manifest.json README.md 'standalone/*.md'
git commit -m "perf: 增加新版完整Search与优化CPU的同期配对测试"
git push
```

黄区无需重新编译或安装，逐条执行：

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -o pipefail
```

```bash
python3 standalone/benchmark_compact_search_cpu.py --input results/search_256k_q8_input1 --acceptance results/search_topk_compact_256k_q8_pair1 --output results/search_compact_cpu32_256k_q8_pair1 --cpu-config 32:32 --device 4 --rounds 6 --warmup 5 --repeats 30 2>&1 | tee results/search_compact_cpu32_256k_q8_pair1.log
```

设备4若被占用就改为空闲同规格卡。前置报告可以来自另一张同规格卡，其旧时间不作为样本；本轮所有NPU进程使用命令指定的同一张卡。结果目录已存在就换名字，不覆盖。

默认使用已编译的`results/search_benchmark_host_build/bm25_cpu_search_benchmark`、同目录`bm25_npu_search_benchmark`，CPU五批入口为`results/cpu_index_search_build/bm25_cpu_index_search_runner`；可通过`--cpu-runner/--npu-runner/--cpu-diagnostic`明确指定。程序会核对实际编译源码，不能拿旧或其他程序绕过检查。

成功状态为`compact_search_cpu_npu_paired_and_all_checks_passed`，仅代表实跑和所有结果核对完成。是否达到NPU更快的目标，以本轮A/B各轮中位数及p95为准，不将历史值拼成加速比。黄区文件留黄区，只分享允许的末尾汇总或第一处错误。

## 对应代码与当前完成程度

`validate_selected`只读复核已验收新组合及其完整结果，排除错误索引/旧程序/安装变更；`cpu_preflight`运行本次CPU五批检查并直接对照既有NPU输出；`run`生成本轮查询文件，调用已有`bench.run_one`交替启动CPU/NPU计时，核对四个完整检查点，再生成A/B汇总。评分上限仍由已验收入口限制：本对照整批最多64词，当前Q8共24词；不代表已支持任意长查询/大批次。

本地74项来源、报告篡改、查询/输出/安装变更、实际线程不足、计时边界、无效传输和失败停止检查通过；特意验证CPU获胜和NPU获胜均能如实保存结果。测试使用真实小输入及独立公式，但可执行文件、安装/前置证明、目录别名和子进程是明确的模拟，不能作为CANN编译或CPU/NPU实际性能证据。本轮实际同期对照后续已通过；当前继续BM25S入口的目标Numba验收。
