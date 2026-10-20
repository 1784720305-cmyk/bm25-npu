# 完整Search：预热、CPU配置扫描和六轮配对

## 现在已经知道什么

用户黄区实机日志确认：正常64K/Q1和256K/Q8的原始词编号Search联合验收均通过。每端A/B/A/空/B五次实际调用、完整文章分数、局部候选、严格排名、重复字节及索引复用通过。

| 正常负载 | CPU正常查询整批诊断(ms) | NPU正常查询整批诊断(ms) |
|---|---|---|
| 65536篇文章、Q1、K10 | 0.019269～0.024717 | 0.080761～0.130526 |
| 262144篇文章、Q8、K10 | 0.412229～0.486419 | 0.600831～0.717005 |

这些范围只包含A/B，不含空查询。在这些记录中NPU更慢；它们不是预热后正式统计，不能用最小值相除宣称稳定倍数。Q8整批时间不能当单查询延迟。

256K/Q8实际整库索引67,031,608字节，一次加载/上传；每批3词×8查询的逻辑输入168字节，Top-K640字节。消除了旧的逐查询展开输入，仍然没有证明最终Search加速；不因上传量减少就直接认定瓶颈已消除。

本次只新增独立计时外壳和报告工具。**原CPU/NPU Search头文件、全部算子、旧调用器和样本原字节不改。新计时程序尚未在黄区编译执行。** 不新增内核，不使用ATC/OM、Faiss、IndexSDK，不引入BM25S运行依赖。

## 用简单的话说明怎么测

输入仍是同一整库索引和CPU已经持有的原始词编号。每个进程先建立一个会话，索引只加载一次，通用容量和合成空查询预启动的冷时间另记。

计时外壳直接调用已验收的`Search`，不重新实现算分或Top-K：

1. 先用实际A、B各运行一次，计时外污染输出，再导出完整结果检查。这两次只建立核对用Top-K，不纳入正式样本。
2. 预热5轮；每轮各调用一次A和B，顺序交替为AB、BA，因此每种查询各预热5次，共10次Search。
3. 正式测30轮，每轮也各调用A、B一次，每种查询各30个样本，共60次Search。正式循环中没有调试污染、文件读写或完整中间结果下载。
4. 每次`Search`返回后检查Top-K与该进程前面的真实结果逐字节一致。此额外正确性检查、样本保存及结果析构在调用计时之外；两端规则一致。`Search`内部必要的输入/结果检查仍在时间内。
5. 正式循环结束后，再污染输出并各执行一次A、B，导出完整分数及NPU局部候选。前后四份完整结果独立核对，前后对应字节也须一致；索引最后核对不变。全部通过后才接受时间记录。

**主指标是调用方主机时钟：调用`Search`前开始，到它返回CPU可用Top-K后停止。** 包含NPU每次词编号整理、两组查询上传、三段接口准备/提交、同步、两组Top-K下载和必要资源清理。还保留旧Search内部时间供核对，要求外部完整调用时间不小于内部时间。不是设备纯内核时间，也不把上传/下载移到计时外。

每个NPU计时样本还检查从Search结果记录的实际应用查询上传字节和返回Top-K字节。这里只说明应用显式拷贝，未声称统计CANN内部全部通信。CPU没有设备传输，相应字节为0，不为CPU加伪造拷贝。

## CPU怎么选、如何降低顺序影响

默认扫描`1:1 8:8 16:16 32:32 64:64`，冒号前是线程数，后是文章分区数。每个候选用三个独立进程复测，扫描顺序轮换。A是原始正常查询，B是保持条数/长度、词编号轮换后的另一批查询。

按各候选的“三次A中位数的中位数”选择最小者，再用该CPU配置与NPU做六轮独立进程配对：第1/3/5轮CPU先，第2/4/6轮NPU先。每个新进程重新初始化，但索引加载/设备初始化和容量准备不计入Search；它们单列冷启动时间。

这个选择规则事先固定，不为了突出NPU而选择更慢的CPU。B单独统计，不把A选出的配置称作B的最佳。也不把默认扫描范围称为整机最优：服务器是128物理核/256逻辑CPU，可按需要用`--cpu-config`增加`128:128`，或检查`16:8`等分区。扫描时保存实际OpenMP团队数、继承的CPU亲和性、可见配额和线程环境变量；这些记录不证明CPU/NPU独占，也没有自动完成NUMA绑核优化。

输出分别保留A/B每次时间、中位数、p95（nearest-rank）、平均时间以及查询吞吐。吞吐按总查询数除以Search调用时间总和，排除调用之间的验收代码；不是整个脚本的服务吞吐。六轮再列各轮中位数和逐轮CPU/NPU比值，**不把若干段中位数相加，不把中位数的中位数叫全部样本的中位数，不声称统计显著性**。

## 黄区先编译新增外壳

提交推送后逐条执行，失败就停；原来的CPU/NPU程序和三个算子不需要重编或重装。

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -o pipefail

cmake -S npu_prototype/search_benchmark_host -B results/search_benchmark_host_build -DCMAKE_BUILD_TYPE=Release -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 -DBM25_VENDOR_DIR=/test/w50066029/bm25-npu/results/custom_opp/vendors 2>&1 | tee results/search_benchmark_configure1.log

cmake --build results/search_benchmark_host_build -j 2>&1 | tee results/search_benchmark_build1.log
```

生成两个独立程序`bm25_cpu_search_benchmark`和`bm25_npu_search_benchmark`，CPU使用与已验收基线相同的`-O3 -march=native -ffp-contract=off -fno-fast-math`/OpenMP；NPU主机使用原Search相同的`-O2 -ffp-contract=off -fno-fast-math`。自动记录每个程序实际编译的CMake/源文件/共享头文件SHA256，运行时要求一致。

## 第一步：64K单查询正式复测

直接复用刚刚通过的输入与报告，不重建语料：

```bash
python3 standalone/benchmark_search.py --input results/search_64k_q1_input1 --acceptance results/search_64k_q1_cpu_npu_check1 --output results/search_64k_q1_benchmark1 --device 2 --shards 32 --cpu-config 1:1 8:8 16:16 32:32 64:64 --scan-rounds 3 --rounds 6 --warmup 5 --repeats 30 2>&1 | tee results/search_64k_q1_benchmark1.log
```

`--acceptance`必须是该输入已经通过的正常规模新Search联合报告，不能换成旧resident或展开数组计时报告。程序复核其源代码、旧实际调用器、日志、原始查询文件、40份输出、会话记录和完整结果；之后调用的是新的计时外壳。旧报告保留，不将其中时间加入正式样本。

## 第二步：64K实验通过后，测256K、8查询

```bash
python3 standalone/benchmark_search.py --input results/search_256k_q8_input1 --acceptance results/search_256k_q8_cpu_npu_check1 --output results/search_256k_q8_benchmark1 --device 2 --shards 32 --cpu-config 1:1 8:8 16:16 32:32 64:64 --scan-rounds 3 --rounds 6 --warmup 5 --repeats 30 2>&1 | tee results/search_256k_q8_benchmark1.log
```

默认每个工作负载15次CPU扫描进程，加上12次配对进程，共27次独立测量。阶段有进度输出，正式Search循环内不打印。256K/Q8会保存约0.91GB完整分数检查点，另加候选/日志/JSON；保存都在Search外，不把这个磁盘量算进查询延迟。历史目录不覆盖，重跑请改输出名。设备2需可用；不指定被他人占用的设备4，不自动抢占或改系统环境。

成功状态为`complete_search_cpu_scan_and_paired_rounds_passed`。终端会打印选择的CPU配置、A/B各轮CPU/NPU中位数和NPU较快的轮数，完整p95及原始样本在各轮报告里。只分享黄区允许的这些终端汇总或首个错误，文件留黄区。

## 关键函数与下一步判断

- `bm25bench::Measure`：共同的AB/BA预热与正式循环；CPU和NPU的计时起止语句相同。调用原`Search`，没有缓存答案代替计算。计时前真实Top-K仅用于检查，不传给算子。
- `check_session`：核对完整边界、冷启动、正式样本数量、每次应用传输字节、编译指纹和交替次序，拒绝无效或缺失计时。
- `check_output/check_checkpoints`：用同输入FP32公式和原始语料独立公式检查全分数，检查全部文章的严格排名、NPU局部候选和选中分数原字节，前后结果必须一致。
- `run_one`：每个CPU配置或NPU轮次启动新进程，失败即停，无CPU替代NPU。
- `benchmark_search.run`：先按固定规则选CPU，再交替配对、直接核对CPU/NPU四个检查点，保留全部时间记录。资源/来源/安装产物改变则停止，不输出成功比较。

本机55项验收包含真实主机数据准备和假程序/模拟子进程，不是新C++编译、设备运行或性能证据。原规模工具56项、CPU工具67项、NPU Search工具128项模拟回归通过；Python3.9语法另行检查。已有两类正常规模实机正确性来自用户日志，与本机模拟测试分开记录。

若正式结果仍是NPU慢，下一步对这条**新常驻索引Search链**采集算分、局部Top-K、合并及主机传输/准备成本，再按占比选优化。不能沿用旧四段采集直接认定新链的瓶颈，也不因950规格强就跳过实测。原版BM25S的同边界适配和Numba实机检索仍待补，不能拿当前CPU/NPU报告冒充BM25S比较或最终任务达标。

建议commit备注：`feat: 增加完整Search预热计时与CPU配置配对基准`
