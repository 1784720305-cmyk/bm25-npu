# Radix候选收集独立实验

本版代码已准备，尚未在CANN编译、安装或设备验收。原Radix仍是主对照；零分汇总版已在256K文章/8查询的六轮同期对比中全部更慢，保留作负收益证据。没有把原版的实机结果记为本版结果。

## 先理解输入、处理和输出

输入仍是一行行FP32文章总分及K，K在1..min(文章数,32)。不改变贡献计算、累加、分块数或精度。

原Radix用三轮整数直方图找到准确第K名的分数门槛。之后原版每128篇文章做一次协作计数，8192篇要做64次。本版仅替换门槛之后的收集：

1. 每线程扫描自己负责的文章，数出严格高于门槛的篇数，并记住第一篇门槛同分文章编号。
2. 128线程统一计数一次，为严格高分文章分配互不重叠的位置；有这类文章的线程才回读写入，找到最后一篇就停。
3. 每线程的同分文章编号天然递增。每次选128个名单头部中最小的编号，只推进胜者；选够所需篇数后停，不额外寻找下一篇。
4. 选出的K篇仍用原32槽排序网络排序，输出原FP32分数字节和INT32文章编号。

输出规则仍为分数降序、数值同分编号升序；+0和-0比较时同分，输出保留原符号位；正次正规数不能当零。非法结果由调用器拒绝，不用CPU答案替代。

代价是部分分数回读、四个warp之间的头部归约、等待及寄存器使用。额外逻辑读取的模型上界为(K-1)*ceil(D/128)，不是内存事务或耗时。64次协作改为一次计数加至多K次头部合并，不能直接换算成加速倍数。短行、稠密数据、K=32也可能更慢，必须测。

验收分四层：代码与本地模型检查；CANN编译/安装；独立设备25组及四段六数组正确性；普通性能轮换比较。前三层通过也不代表性能达标。

## 关键代码怎样解释

| 文件/函数 | 作用及不变量 |
|---|---|
| `npu_prototype/topk_radix_collect/op_kernel/bm25_top_k.cpp` 的 `RadixTopKSimt` | 三轮门槛计算逐字保留原Radix；只替换收集段。每线程仅保留计数、位置和一个同分头部，无128份K项名单 |
| `PrefixPair` | 复用原128线程整数前缀计数；收集整行调用一次，为严格高分候选分配互斥槽位。原门槛阶段的调用不变 |
| `MinimumHead` | 每32线程用XOR交换求最小编号，再通过明确分配的UB合并4个头部；写后、读后都有全组屏障，耗尽线程也参与 |
| `RadixTopKSimt` 的 `privateStrict` 和 `equalHead` | 严格高分回读按自己的独占槽位写；同分只推进胜者，保证挑选较小编号，不能按atomic到达顺序抢名额 |
| 新 `op_host` 两个文件 | 与原Radix字节相同：分核及8512字节算法UB不变，不增加用户GM候选空间。CANN系统工作区仍以实际接口返回及验收报告为准 |
| `validate_topk_radix_collect.py` | 独立版本标识，核对新工程源码、实际头文件、API库和设备目标文件；交替A/B、每次覆写输出、C++完整排序及Python独立核对 |
| `topk_radix_collect_host/CMakeLists.txt` | 新目标名和独立vendor，复用已验证的原Radix C++调用器源码；构建时仍使用本版实际生成头文件 |
| `prepare_sharded_runtime.py` | 新版25组全部通过才允许建立独立四段运行别名，不接受冒烟、旧版或零分汇总验收报告 |
| `compare_radix_versions.py --candidate-kind radix_collect` | 原版与本版在同卡、同输入、同四段调用器上轮换；每轮六数组及跨轮字节一致才汇总时间 |

复用了已在项目实际编译过的SIMT原语，但新组合的编译、屏障/可见性及性能仍须实机验证。本版参考来源和选择理由见 [修订路线](Radix候选收集修订路线.md)；不引入IndexSDK、Faiss或其他算法库运行依赖。

## 第一步：仅生成、编译新算子

请先提交并在黄区拉取本次代码，再执行。以下命令在黄区Linux执行。某一步失败就停止，不继续运行后面的步骤；已存在的结果目录用新名称，不删除旧证据。

```bash
cd /test/w50066029/bm25-npu
source /usr/local/Ascend/cann-9.2.0/set_env.sh
python3 verify_package.py
set -o pipefail
python3 standalone/create_project.py --operator topk_radix_collect --compute-resource ai_core-ascend950 --compute-unit ascend950 --msopgen-python python3
```

只有生成成功后执行：

```bash
cd /test/w50066029/bm25-npu/results/Bm25TopKRadixCollect
bash build.sh 2>&1 | tee ../radix_collect_build1.log
```

成功标准是构建退出码为0，确有新安装包，而非仅出现生成源码的日志。第一次先到这里，核对构建日志后再推进安装。

## 后续第二步：安装并核对实际接口

仅在第一步成功后使用此节，不覆盖原Radix vendor。

```bash
cd /test/w50066029/bm25-npu/results/Bm25TopKRadixCollect
bash build_out/custom_opp_openEuler_x86_64.run --install-path=/test/w50066029/bm25-npu/results/custom_opp
```

安装成功后读取新vendor实际头文件：

```bash
cat /test/w50066029/bm25-npu/results/custom_opp/vendors/bm25_topk_radix_collect/op_api/include/aclnn_bm25_top_k.h
```

核对它是否仍是已确认的两阶段接口，参数依次为scores/k/topScoresOut/topIdsOut/workspaceSize/executor，以及workspace/workspaceSize/executor/stream。检查器会严格核对完整声明；不同就停止，不改检查器强行放行。

## 后续第三步：独立调用器和正确性

接口核对后，在新构建目录配置：

```bash
cd /test/w50066029/bm25-npu
cmake -S npu_prototype/topk_radix_collect_host -B results/topk_radix_collect_host_build -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 -DBM25_OPP_ROOT=/test/w50066029/bm25-npu/results/custom_opp/vendors/bm25_topk_radix_collect 2>&1 | tee results/radix_collect_host_configure1.log
```

配置成功后编译，再冒烟，各步失败即停：

```bash
cmake --build results/topk_radix_collect_host_build -j 2>&1 | tee results/radix_collect_host_build1.log
```

```bash
python3 standalone/validate_topk_radix_collect.py --suite smoke --output results/radix_collect_smoke1 --device 2 2>&1 | tee results/radix_collect_smoke1.log
```

冒烟通过后完整验收：

```bash
python3 standalone/validate_topk_radix_collect.py --suite all --output results/radix_collect_topk_validation1 --device 2 2>&1 | tee results/radix_collect_topk_validation1.log
```

25组覆盖K=1/10/32、K=全部文章、全零/正负零/次正规数/最大有限数、同分跨128边界、单线程同分流、无后继头部、行尾候选、多行及8192/8193列等。每组至少5次实际交替调用，A/B都重复；C++逐次检查全编号和原分数字节，Python独立核对A/B最后结果。报告必须为`radix_all_unit_device_and_full_sort_passed`，并且本版variant、源码/目标文件及25项完整匹配；仅打印成功或冒烟不够。

## 后续第四步：四段正确性

完整独立验收通过后再建立本版运行目录。其余三个算子及原四段调用器保留：

```bash
python3 standalone/prepare_sharded_runtime.py --topk-implementation radix_collect --output results/radix_collect_sharded_runtime
```

先小样本，后目标数据；各步核对通过再继续：

```bash
python3 standalone/run_sharded_pipeline.py --input results/pipeline_input --output results/radix_collect_small_s2_check1 --runner results/radix_sharded_host_build/bm25_sharded_benchmark --vendor-dir results/radix_collect_sharded_runtime/vendors --shards 2 --device 2 --warmup 2 --repeats 3 2>&1 | tee results/radix_collect_small_s2_check1.log
```

```bash
python3 standalone/run_sharded_pipeline.py --input results/normal_256k_q8_batch1 --output results/radix_collect_256k_q8_s32_check1 --runner results/radix_sharded_host_build/bm25_sharded_benchmark --vendor-dir results/radix_collect_sharded_runtime/vendors --shards 32 --device 2 --warmup 2 --repeats 3 2>&1 | tee results/radix_collect_256k_q8_s32_check1.log
```

```bash
python3 standalone/compare_compute_outputs.py --input results/normal_256k_q8_batch1 --cpu results/cpu_256k_q8_scan1/r1_s8 --npu results/radix_collect_256k_q8_s32_check1 --report results/radix_collect_256k_q8_s32_cross1.json 2>&1 | tee results/radix_collect_256k_q8_s32_cross1.log
```

CPU路径使用此前已通过的8线程/8文章分区结果；如果目录不在，不跳过核对，先确认实际结果位置。四段检查同时覆盖贡献、总分、局部候选、最终排名以及首末重复六数组。此阶段3次计时只作诊断。

## 后续第五步：决定保留还是放弃

先关闭性能采集，再与原Radix做同卡六轮交替，而非与已更慢的零分汇总版比较：

```bash
python3 standalone/compare_radix_versions.py --candidate-kind radix_collect --input results/normal_256k_q8_batch1 --output results/radix_collect_vs_original_256k_q8_pair1 --device 2 --shards 32 --rounds 6 --warmup 5 --repeats 30 2>&1 | tee results/radix_collect_vs_original_256k_q8_pair1.log
```

输入、四段调用器、其他三算子、分块及注册/分核源码一致，仅换局部Top-K内核。每版本使用独立进程，轮换谁先跑；保存全部30个样本、每轮中位数/p95、配对比值、库/设备对象来源及六数组。计时包含接口准备、四次提交和同步，不含初始化、分配、传输、文件、输入整理和调试下载。

目标256K/Q8若仍无可重复收益，不因模型减少同步就切主线，不继续叠加此微调；回到稀疏分数流程和累加。若有清楚收益，再在64K/Q1、256K/Q1及稠密、K=32/短行补验退化；单条延迟与批量吞吐分开。普通计时收益确定后才按需采集定位，不能用采集结果代替正式对比。之后仍须与较优CPU配置同期比较、原版BM25S、含传输及完整查询验证。

提交备注建议：`feat: 增加Radix候选收集独立实验及设备验收`
