# CPU 性能回归核查

2026-11-04 的 A5 正确性、SDK 资源共享及 NPU 优于本会话 CPU 的门槛已通过。这些门槛不检验 CPU 相对旧版是否保持速度。NPU A/B 仍约 2 ms，而该验收会话的较快 CPU 逐轮中位数约 6～21 ms。后续补充了同机、同输入、同放置条件下的旧新对照，结果与适用范围见文末。

## 已确认的差异

用户回传的两套实际 `flags.make` 均使用 `/usr/bin/c++`，最后生效的优化级别均为 `-O3`，并保留 `-march=native -ffp-contract=off -fno-fast-math -fopenmp`。新构建还通过 `bm25_common` INTERFACE 传播了：

```text
-D_FORTIFY_SOURCE=2
-fstack-protector-all
-fno-strict-aliasing
-Wreturn-type
-fno-common
-fPIC
```

链接也增加了 relro/now/noexecstack 选项。因此本次改动除了目录调整，还改变了 CPU 的编译策略。[GCC 优化说明](https://gcc.gnu.org/onlinedocs/gcc-10.3.0/gcc/Optimize-Options.html)描述了类型别名优化；`-fno-strict-aliasing` 会关闭这一优化依据，不能假定它对热循环没有影响。[栈保护说明](https://gcc.gnu.org/onlinedocs/gcc-10.3.0/gcc/Instrumentation-Options.html)说明了额外的运行时保护代码。各项实际开销必须测量。

对 `db6559e` 和 `9a99bbc` 的 CPU Search、目录 Search、分区 Top-K、堆插入及计时模板进行函数体 token 对照，忽略格式和注释后相同。该检查不证明生成的机器码、所有调用位置的优化决策或整体性能相同。记录见 [源码核查](../evidence/cpu_regression_source_audit_20261104.json)。

服务器为双路 AMD EPYC 9575F，共 256 逻辑 CPU、两个 NUMA 节点。主进程允许使用 0～255；源码没有显式固定 OpenMP 线程位置，原测量的 OpenMP 环境未记录，不能据此断定历史线程实际分布。CPU/内存放置、负载及编译策略都需要控制。

## 一次受控实验

`tools/compare_cpu_builds.py` 比较三组目录 CPU：

| 组 | 来源 | 用途 |
|---|---|---|
| old_binary | 原 `final_host_build` 二进制 | 当前相同环境下的旧版参考 |
| new_binary | 已通过 A5 的 `9a99bbc` 二进制 | 当前异常版本 |
| new_source_old_flags | 隔离提取 `9a99bbc` 源码，仅在临时副本还原旧 CPU 编译策略后构建 | 区分源码变化与编译策略变化 |

实验不修改工作区 C++、生产 CMake 或既有二进制。临时编译策略用于诊断，不能作为已批准的生产安全策略。

每组两次，顺序为 old/new/new_old_flags/new_old_flags/new/old。每个进程保持 Q8/K10、32 线程/32 分区、5 个预热周期及每个查询集 30 个计时样本。固定 CPU 0～31 和内存节点 0，并校验它们为允许使用的 32 个不同物理核。`numactl` 绑定失败会停止，不会静默改成未绑定测量。

所有 before/after 的全分数、Top-K 分数及 ID 共 12 个文件必须在各组间长度及 SHA256 一致，实际 OpenMP team 必须为 32。每次子进程运行前后由父进程记录负载、CPU 频率/调频器、OpenMP 环境和资源限制信息；子进程放置以绑定探测和实际命令为准，父进程 affinity 不代表子进程 affinity。同时记录真实编译参数、依赖和二进制指纹。

在 A5 依次执行以下命令：

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
python3 tools/compare_cpu_builds.py --before-build /test/w50066029/bm25-npu/results/final_host_build --after-build /test/w50066029/bm25-npu-indexsdk/results/indexsdk_a5_acceptance_02/build --input /test/w50066029/bm25-npu/results/large_scale_close1/inputs/10000000 --output results/cpu_regression_01 --cpus 0-31 --numa-node 0
```

结果目录必须是新的；失败结果保留，下次更换 `_02`。默认每个试验的全分数大文件在完成指纹和等价性核对后删除，只清理本次新建试验目录中的已知 checkpoint 文件，保留 JSON、日志及全部 SHA256。原有 A5 数据和验收结果保持原样。添加 `--keep-checkpoints` 可保留本次大文件，此时磁盘预算按六个进程计算。

最后打印 `REPORT` 和 `LOG BUNDLE`，回传该 JSON/ZIP。两次试验/组仅用于定位明显差异；需要结合成对耗时、负载和输出等价性解释，不直接替代完整性能验收。

若新源码恢复旧策略后接近旧程序，而新程序仍较慢，编译策略的影响得到支持；若三组在绑定后均恢复且相近，放置/负载是更强的候选解释；若旧策略的新源码仍持续慢于旧程序，应继续检查源码、编译器和生成代码。这些是判别思路，两次试验不一定足以排除运行噪声。

## 已回传的三组对照

`results/cpu_regression_01` 的状态为 `completed_with_bit_identical_outputs`。用户回传了六次终端耗时及报告中的绑定、二进制指纹和环境摘要，归档于 [对照记录](../evidence/cpu_regression_a5_comparison_20261104.json)。原始 JSON/ZIP 尚未传到本地。新版二进制 SHA256 与已归档 A5 验收完全匹配；绑定探测成功，CPU 0～31、内存节点 0，OpenMP 32 线程、close/cores。

以下数字是每组两次进程中位数的中位数，单位为每批 Q8 的 ms：

| 组 | A | B |
|---|---:|---:|
| old_binary | 5.473 | 5.552 |
| new_binary | 5.753 | 5.785 |
| new_source_old_flags | 5.395 | 5.390 |

新版汇总比旧版高 5.11%/4.19%，但同一新版两次进程的中位数从 6.115/6.138 ms 降到 5.391/5.432 ms，组内差异达 13.45%/13.01%。恢复旧参数的结果接近旧程序，第二次保留全部参数的新版也接近该范围。因此尚不能把汇总差异当成参数的固定开销；此前数倍的时间差没有在本次共同条件下复现。

各进程前后的 1 分钟 load 为 211.76～231.82，5 分钟为 252.2～262.59，15 分钟为 272.87～277.19。[Linux load 定义](https://man7.org/linux/man-pages/man5/proc_loadavg.5.html)包含可运行及不可中断 I/O 等待任务，不能换算为 CPU 利用率，也不能仅凭总负载认定所选核受到争用。这表明存在较高系统负载背景，当前试验没有证明获得了空闲、独占的执行资源。[CPU affinity](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html)限制本进程可运行的位置，不会自动排除其他任务。

频率快照多数约 5 GHz，个别采样较低；这些运行前后快照不能证明计时期间持续降频。PSI 与根 cgroup-v2 字段为 null，含义是相应路径没有读到数据，不能据此排除配额或节流。

本轮保留结构、生产参数和原验收结果。剩余差异需要在较空闲、可预留 32 个物理核的窗口复核，预留时也应考虑这些物理核的 SMT 兄弟线程。届时复用上述命令，将输出改为 `results/cpu_regression_02`；无需重跑 NPU 验收。不要为获得独占环境自动停止其他作业或修改系统策略。若稳定条件下差异仍重复出现，再逐项隔离编译参数；若差异消失，则保留当前参数并固定基准运行条件。
