# SIMD 候选容量 1024 / 2048 / 4096 对照

当前状态：修复后的 v2 打点包与诊断主机已在 A5 完成构建、真实头文件 clang-tidy、六组边界检查，以及三个输入、三个容量的完整六轮测量。54 个测量单元、正常/打点包 Top-K 位一致性和 v2 时间戳检查均通过，结果见下表。4096 在 SciFact B 获益明显，但原 8192 稀疏输入有中位/P95 退化；**默认继续保留 1024，4096 保留为候选**。旧 v1 阶段值仍作废，不混入本轮。

## 变量与布局

`standalone/selector_capacity.py build` 从源码锁定的 fused 实现生成三个容量。正常包只调整候选容量及对应 UB 布局/声明；1024 正常包的 kernel 与当前基线逐字节相同。每个容量另生成一个诊断包，共六个新 vendor，在新的 `results/` 目录内安装。保留已有 vendor、数据、主机程序和 Merge 包；所有运行在独立子进程中设置本次实际 vendor 路径。

| 容量 | 选择 scratch 字节 | 完整 TBuf 字节 | 动态 UB 声明字节 |
|---|---:|---:|---:|
| 1024 | 16736 | 50848 | 65536 |
| 2048 | 24928 | 59040 | 65536 |
| 4096 | 41312 | 75424 | 98304 |

4096 不能沿用 64 KiB 声明。三者保留原来的 DCache 至少 32 KiB 与编译器预留 8 KiB 的独立检查；硬件实际 UB 不足时停止，不缩减保护区。直方图最大计数 4096 仍在 uint16 范围内。输入正分总数超过容量时，继续使用通用选择路径，不能截断候选。

同源 SciFact 性能输入每批 Q8，按本地原始准备记录推导的 SIMD 条件覆盖如下。它们是输入统计，不是设备执行时间：

| 容量 | A 符合条件的查询叶块 | B 符合条件的查询叶块 | 总计 |
|---|---:|---:|---:|
| 1024 | 0/8 | 3/8 | 3/16 |
| 2048 | 2/8 | 5/8 | 7/16 |
| 4096 | 7/8 | 8/8 | 15/16 |

查询词数、查询批量、K、索引字节和文档顺序不变。扩大覆盖范围可能增加 UB、扫描和同分收集成本，覆盖率不能替代计时。

## 两种测量

- **完整 Search**：使用无打点正常包和现有 `indexsdk_alloc_host_01/build/bm25_large_search_benchmark`。计时边界仍为 CPU 已有原始词 ID 到 CPU 拿到 Top-K；保留独立完整分数、叶/合并层、严格 ID、保护区、索引回读和 A/B 重复位检查。首次上传、预热、诊断、文件 IO 不计入 Search。
- **阶段诊断**：使用单独的打点包和 `capacity_profile`。诊断模式 3 将评分与候选追加、候选 Pack 与阈值/收集分开，用真实索引和查询执行；不使用预计算分数重放替代真实评分。每次比较打点/正常模式的完整 Top-K ID 和分数位，并检查 trace 保护区和设备端容量标记。新模式只存在于生成的诊断包。

阶段输出是每个查询叶块的 `GetSystemCycle` 原始 ticks，包含同步及打点边界开销；分离 Pack 的额外调用也影响诊断。**不能将多个核/叶块的耗时相加冒充整批 Search，也不能将诊断值与正常 Search 相减推导其他阶段**。报告同时保留全部原始任务时间戳。官方 [GetSystemCycle 文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha001/API/ascendcopapi/atlasascendc_api_07_0282.html)说明了流水同步要求及其所覆盖型号的换算；当前没有独立确认 A5 的计数器频率，因此不套用旧型号的 50 MHz 输出微秒。首次 A5 构建也会检验该接口能否用于当前 CANN/A5。

每个容量在六轮中各占两次第一、第二、第三位置，使用全部六种排列。每轮交替正常/诊断的先后顺序；各进程预热 5 次，A/B 各测 30 次。阶段汇总是同一 case 的任务时长分布；完整 Search 汇总是批延迟，两者分开比较。报告记录同轮延迟变化、P95、胜出轮数、输入及安装包完整指纹。已有原版 BM25S 结果作为 CPU 背景保留，本轮只比较 NPU 容量，不重建或替换 CPU 基准。

## 2026-11-06 时间戳修复

旧打点每任务写四个 int64，即 32 字节；相邻核的记录共用 64 字节缓存行。使用 `GlobalTensor::SetValue` 又未做 DCache 写回处理，存在随机覆盖或主机读到未更新内容的风险。[官方 SetValue 约束](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0007.html)明确说明了写回和多核地址间距要求。这两处源码缺陷足以导致缺失时间戳，与本次报错一致；失败当时的原始时间戳未回传，仍需修复后的设备复验确认。

v2 每个任务独占 64 字节，通过与普通结果相同的 SIMT GM store 路径写出，不再使用 Scalar `SetValue`。记录包含开始/评分完成/选择完成时间、容量、输入调用序号与回显、任务号及版本。主机保留严格递增、序号、任务、容量、格式和保护区检查；失败时另存 `.failure.json`，不放宽判定。旧四字段记录不能作为 v2 数据接收。

`capacity_a5_01` 中已完成的无打点 Search 结果保留。首轮 SciFact B 的 4096 相对 1024 为 0.046219/0.074181 ms，耗时低约 37.70%；A 为 0.0865995/0.08658 ms，接近。只有一轮且稀疏回归未完成，不作采用结论。旧阶段值即使通过旧程序检查，也不能用于阶段优化结论；新六轮不会混入它们。

## v2 A5 六轮结果

`results/capacity_a5_02` 已完成。用户提供全部 54 份单元摘要、108 条测量命令及最终 `COMPARISON`，已独立重算排列、样本数、QPS、同轮差值、胜出轮数和全部汇总。原始时间戳及独立报告文件未随终端摘要完整传回，阶段和正确性检查通过的依据为已完成运行器的严格检查流程。首次原包构建及本次三个打点包构建日志均已回传。[本轮记录](../evidence/capacity_a5_20261106.json)。

下表中位批延迟单位为 ms，每批 Q8/K10；数值为六个轮次中位数的中位数。P95 变化是六个轮次 P95 的中位汇总变化，不是合并原始样本后重新计算的 P95。

| 输入/查询组 | 1024 | 2048 | 4096 | 4096 中位变化 | 4096 P95 变化 |
|---|---:|---:|---:|---:|---:|
| SciFact A | 0.085849 | 0.085601 | 0.085799 | -0.06% | +0.28% |
| SciFact B | 0.072905 | 0.072846 | 0.045128 | -38.10% | -37.60% |
| 稀疏 8192 A | 0.030148 | 0.030581 | 0.031192 | +3.46% | +8.03% |
| 稀疏 8192 B | 0.030909 | 0.031450 | 0.031772 | +2.79% | +5.60% |
| 稀疏 65536 A | 0.040035 | 0.039009 | 0.039184 | -2.13% | -3.08% |
| 稀疏 65536 B | 0.040826 | 0.039832 | 0.039915 | -2.23% | -4.58% |

4096 在 SciFact B 的中位与 P95 均六轮胜出，同轮中位变化为 -37.67%～-38.60%，这组收益稳定。SciFact A 只有 2/6 中位胜出，不能以 -0.06% 宣称有效加速。2048 的 SciFact A/B 中位变化仅 -0.29%/-0.08%，各 3/6 胜出，没有明确完整 Search 收益。

8192 稀疏输入的 4096 中位只有 2/6 胜出，A/B 同轮变化范围分别约 -1.76%～+6.37%、-3.22%～+6.84%；不能说每轮都慢，也不能据此抹掉汇总 +3.46%/+2.79% 以及 P95 +8.03%/+5.60% 的退化。这些是本轮实测代价，尚未独立归因到 UB、编译布局或调度。2048 同样在该输入出现中位约 +1.44%/+1.75%、P95 约 +1.01%/+6.40% 的变化。65536 稀疏输入则有约 2%～3% 汇总收益，表明容量收益依赖工作负载。

阶段诊断解释了为什么覆盖率不能代表完整 Search：SciFact A 的 4096 Top-K 任务中位从约 32921 降到 5554 ticks（-83.13%），但任务 P95 仍约 32556 ticks；B 的中位从约 32501 降到 4509 ticks，P95 也从约 33023 降到 6033 ticks。2048 的 B 虽有类似的任务中位下降，任务 P95 仍约 32997 ticks，对应的完整 Search 也未呈现相同收益。阶段单位仍为原始 ticks，不能跨核相加或换算成未经确认的微秒。

用户随后只读汇总了已有六轮逐查询记录，每查询共 180 个样本。A 的第 8 条查询有 **4399 个正分候选**，是 4096 下唯一剩余的通用路径：其 Top-K 中位从 32990 变为 32553 ticks（约 -1.32%），评分/追加从 33034.5 变为 33362.5 ticks（约 +0.99%）；两阶段在 4096 的 A 组均最慢。其他 7 条 A 查询的 Top-K 已降到约 4491.5～7306.5 ticks。这个尾部任务与整批 A 几乎不变相符，不能用大多数任务的中位收益替代整批收益。B 在 2048 下仍有第 1/6/7 条通用路径；4096 下 8 条均进入快速路径，完整 Search 才稳定获益。完整原始时间戳未下载，逐查询数值来自用户汇总，不冒充本地独立上板测量。

本轮不更改默认容量，也不需要为归档重跑。下一步优先优化剩余通用选择路径，以 A 的第 8 条查询作定位样本，同时保留稀疏回归；不直接全局扩大容量。采用任何策略前，还需更广的真实查询覆盖和相同输入的完整 Search/稀疏回归验证。

## A5 复现（已完成，无需为归档重跑）

只复制代码，不带终端提示符。新目录如果已存在，应另取名称；失败保留日志，不删除原目录。

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 -u standalone/selector_capacity.py build --output results/capacity_build_02 --reuse-normal-build results/capacity_build_01 --template /test/w50066029/bm25-npu/results/Bm25FusedScoreTopKUbFix1 --vendors /test/w50066029/bm25-npu/results/final_runtime/vendors --clang-tidy /test/w50066029/bm25-tools/bin/clang-tidy --jobs 8
```

本机已有成功的 `capacity_build_01`，上述命令先核对三个正常包的源码及安装指纹，再复用它们；只构建三个新打点包及两个诊断主机程序，并运行真实头文件 clang-tidy。旧包、旧主机和旧日志全部保留。成功状态为 `experimental_packages_and_host_built_device_pending`，报告须含 `profile_trace_contract: simt_gm_cacheline64_nonce_v2`。首次在其他机器完整构建时，不传 `--reuse-normal-build`，将生成六个包。构建失败时回传 `results/capacity_build_02/build.json` 与失败日志末尾。

构建成功后，在预留 CPU 0～31 及 SMT 兄弟核的窗口执行。`numactl` 只负责放置，不负责预留：

```bash
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
numactl --physcpubind=0-31 --membind=0 python3 -u standalone/selector_capacity.py run --build results/capacity_build_02 --host-build results/indexsdk_alloc_host_01/build --output results/capacity_a5_02 --device 5 --workload scifact=results/scifact_data_01/performance --workload sparse8192=/test/w50066029/bm25-npu/results/score_bitmap_scales1/inputs/8192 --workload sparse65536=/test/w50066029/bm25-npu/results/score_bitmap_scales1/inputs/65536
```

先对全部六个包执行扩展选择检查：原有边界外，新增 2047/2048/2049、4095/4096/4097 正分数量，覆盖 K1/10/32、同分、FP32 边界、A/B/A 和保护区。然后对相同 SciFact、原 8192 与 65536 稀疏模拟输入逐轮测量。保留所有轮次，不因某轮慢而删样本。

完成后回传末尾 `COMPARISON:`，以及 `results/capacity_a5_02/report.json`。如果失败，回传该报告和首个失败日志末尾；如有 `.failure.json` 也一起回传。新目录重新执行六轮，保持相同打点格式与顺序，不拼接旧阶段样本；无需重跑已验收的原版千万篇/884 万篇。

是否采用候选，以真实输入的完整 Search 为主，同时检查阶段变化和两个原稀疏输入。Top-K 变快而 Search 不快时仍记录完整差距。若稀疏输入稳定慢 4%～5%，作为实际候选开销处理；不以全机负载笼统排除，也不自动批准。两种候选都无实际收益时继续保留 1024。
