# 定位当前三段Search为什么慢

## 正式基线已经完成，当前没有加速达标

用户黄区普通运行（未开启msprof）已完成两个正常合成负载的配置扫描、预热、六轮进程顺序交替与全部结果核对。以下为原查询A“六轮中位数的中位数”，不是全部原始样本合并后的中位数：

| 负载 | CPU已扫描A最佳配置 | CPU完整Search(ms) | NPU完整Search(ms) | NPU更快轮数 |
|---|---|---:|---:|---|
| 65536文章，Q1，K10 | 16线程/16文章分区 | 0.0146245 | 0.0763145 | 0/6 |
| 262144文章，Q8，K10 | 32线程/32文章分区 | 0.07011525 | 0.579449 | 0/6 |

查询B也均为0/6轮NPU更快。当前NPU实现约需CPU的5.2倍、8.3倍时间，仅限这些输入/配置/环境；不据此判断硬件普遍强弱，也不把8查询整批时间除以8当单条延迟。CPU候选为1/8/16/32/64线程，不称整机最优；原版BM25S同边界/Numba比较仍待补。

当前已实现MT要求的整库原始TF索引常驻＋每次传查询词编号，两端结果正确。**测量完成不等于性能目标完成。** 接下来先查这条新Search链的实际耗时，再按测量选择规模扩展或计算优化；旧贡献/累加四段流程的采集保留为历史参考。

## 输入、处理、输出与验收

输入复用已通过的`results/search_256k_q8_input1`和`results/search_256k_q8_benchmark1`，运行已实际编译通过的`bm25_npu_search_benchmark`。三个算子、Search实现和C++计时外壳全部原字节不改，不需重新编译或重装；本次只增加Python采集入口、只读分析器和测试。

采集程序先核对正式报告来源、当前程序/源码、全量索引和原始查询，再由原计时外壳实际执行一次。只有一个NPU进程，不重跑CPU配置扫描。仍然包括每次查询上传和Top-K下载；无CPU回退或预存答案替代。计时前后完整分数、候选、严格排名、重复字节和静态索引继续检查。

分析器核对实际`op_summary`与`task_time`的设备/Stream/任务ID、算子名、输入形状、工作块和微秒时间，按C++程序顺序拆分：

- 冷启动合成空容量查询：1次Search。
- 计时前A/B检查点：2次。
- 预热5轮，每轮A/B各一次：10次。
- 正式10轮，每轮A/B各一次：20次，即A、B各10个诊断样本。
- 计时后A/B检查点：2次。

因此默认应有35次三段链，共105条自研设备任务，各35条`Bm25QueryScore`、`Bm25TopK`、`Bm25TopKMerge`。必须与实际CSV一致才接受映射；不能看到105行就跳过形状、顺序及时间交叉。正式统计只取中间20次，排除冷启动、预热和四个检查点。缺失/重复/混入旧算子、设备不符、CSV表头不符或正式主机时间匹配异常均停止，不猜轮次。

输出分别统计A/B每段设备中位数/p95、每次设备链跨度、同次采集下完整Search时间、逐次算出的链外时间。使用Decimal解析大时间戳，避免浮点减法丢失微秒精度。

“链外时间”是**对应同一次完整Search时间减去该次三段设备链首尾跨度**，包含查询整理、上传、接口准备、提交/同步、结果下载、框架/采集影响等未分类开销，不直接称CPU耗时、拷贝耗时或空闲时间。不能把两个中位数直接相减，也不能用普通运行约0.579ms减去本次开启采集的设备时间来分摊成本。

API统计表覆盖全程，包括设备初始化、预热、调试、内存分配与释放，因此单独列出，不用其合计计算稳态占比。不由`aiv_vec_ratio`或工作块数推算整卡利用率。

## 黄区执行：先采集256K/Q8

提交推送后逐条执行，失败就停。本次清单更新，不需要运行cmake或重新编译算子。

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -o pipefail
```

沿用此前黄区实际运行成功的msprof参数与Python子进程采集方式：

```bash
msprof --output=results/msprof_search_256k_q8_1 --runtime-api=on --task-time=l1 --ai-core=on --aic-metrics=PipeUtilization --python-path=/usr/bin/python3 python3 standalone/profile_search.py --input results/search_256k_q8_input1 --benchmark results/search_256k_q8_benchmark1 --output results/search_256k_q8_profile1 --device 2 --shards 32 --warmup 5 --repeats 10 2>&1 | tee results/msprof_search_256k_q8_1.log
```

应用结果在`results/search_256k_q8_profile1/measurement/actual`，采集表在`results/msprof_search_256k_q8_1`下。应用成功状态是`profile_search_execution_and_independent_checkpoints_passed`，还需要CSV分析成功，才能判断设备耗时。本工具不因被名为profile的目录包住，就声称真正采集到设备表。

采集和结果检查通过后：

```bash
python3 standalone/analyze_search_profile.py --profile-root results/msprof_search_256k_q8_1 --run results/search_256k_q8_profile1 --output results/search_256k_q8_profile_analysis1.json 2>&1 | tee results/search_256k_q8_profile_analysis1.log
```

分析成功状态为`current_search_profile_rows_and_measured_samples_cross_checked`，终端会分别打印A/B的三段耗时、设备链跨度、采集下完整Search和链外时间。文件留黄区，只分享允许的终端汇总或首个错误。

输入/运行/分析报告均不覆盖已有目录或报告。重复采集请同时将msprof目录、应用输出目录、分析JSON和对应日志编号改为2，保留原始记录。当前示例沿用空闲设备2，不指定被占用的设备4。若CSV有多份匹配文件，按错误提示缩小`--profile-root`到对应PROF目录，不任取第一份。

## 依据采集结果怎么决定优化

先看设备三段本身用了多少时间。若算分占主导，回到索引查找、命中记录读取、清零和完整分数写回等工作；若局部Top-K占主导，才研究缩小扫描范围、复用成熟选择策略或融合，继续保留FP32和严格同分规则。若链外占主导，再增加专门的主机阶段诊断，区分查询上传、接口准备、同步和下载，不能把它们全部归因于PCIe。

无论哪个方向，优化前先解释输入、减少了什么工作、引入什么成本及验收方法。修改后复用当前普通正式基准的同一计时边界与合理CPU配置，正确性通过再做配对比较；采集下的时间只用来定位问题。

关键函数：`profile_search.run`复用当前实际程序和全部结果检查；`validate_record`绑定程序、日志、前后输出和源码；`analyze_search_profile.phases`精确列出35次调用；`analyze`核对两张设备表、所有输入形状和对应主机样本，然后逐次算跨度与链外时间。

本机34项检查使用真实主机数据准备、模拟NPU子进程和合成CSV，覆盖来源、严格排名、计数/形状/设备/时间/重复文件及失败停止。**不是这条新Search的真实msprof采集证据。** 新采集尚待黄区执行，旧普通正式复测已经通过，两者分开记录。

本次建议commit备注：`feat: 增加常驻索引Search三段采集与耗时交叉分析`
