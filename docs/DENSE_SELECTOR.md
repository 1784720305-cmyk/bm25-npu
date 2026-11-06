# 通用选择路径的 SIMD 阈值候选

隔离候选已完成 **A5 CANN 编译、设备正确性和六轮配对测量**：SciFact A/B 的完整 Search 分别降低 **24.43% / 29.37%**，中位延迟与 P95 均六轮全胜。稀疏路径仍有实测代价，暂不替换已验收的 1024 默认包。[回传证据](../evidence/dense_selector_a5_20261106.json)保留全部 36 个单元、配对差值、源码指纹和逐查询摘要。

## A5 六轮结果

用户在 `dense_selector_build_01` 完成两个候选包构建，在 `dense_selector_a5_01` 完成四个正常/打点包的边界检查及 72 条测量命令。三个输入、两个版本、六轮共 36 个单元；正常 Search 与打点包各有 2160 次 A/B 计时调用，打点记录共 57600 条。正常/打点以及候选/基线的 Top-K 字节一致，三个输入各自的输出哈希在所有轮次与版本间一致。

下表的 Search 和 P95 为六轮统计值的中位数；变化为候选相对基线，负值更快。它们是每批 **Q8** 的完整 Search 耗时，不能作为单查询延迟。

| 输入 | 基线 Search ms | 候选 Search ms | Search 变化 | P95 变化 | 配对 Search 胜出 |
|---|---:|---:|---:|---:|---:|
| SciFact A | 0.08533825 | 0.06448900 | -24.43% | -24.90% | 6/6 |
| SciFact B | 0.07237350 | 0.05111925 | -29.37% | -29.70% | 6/6 |
| sparse8192 A | 0.03055850 | 0.03046850 | -0.29% | -2.08% | 2/6 |
| sparse8192 B | 0.03105150 | 0.03132900 | +0.89% | -1.30% | 2/6 |
| sparse65536 A | 0.03985225 | 0.03998475 | +0.33% | +1.10% | 2/6 |
| sparse65536 B | 0.04072100 | 0.04099900 | +0.68% | +3.12% | 2/6 |

SciFact A/B 的配对中位延迟变化范围分别为 **-26.87%～-23.30% / -32.83%～-28.59%**。逐查询摘要确认 A 第 8 条查询的 Top-K 从 **32996.5 降至 10792 ticks（-67.29%）**，评分/追加从 33033.5 到 33202.5 ticks（+0.51%）。此前超过 4096 候选后落入慢路径的任务已改善，这支持本次阈值路径优化的作用。分阶段来自独立打点包，单位为原始 ticks，不能与完整 Search 的毫秒数直接相加。

稀疏代价同时保留：四个稀疏测试项都在 **4/6 轮**的配对 Search 中变慢，配对变化的中位数分别为 **+2.02%、+2.47%、+1.70%、+2.33%**；不能因六轮汇总比值接近零就抹去这些差异。其打点 Top-K 中位统计增加 **4.32%～5.09%**，sparse65536 B 的汇总 P95 增加 3.12%。这些是本轮观察到的代价，不用全机负载笼统排除，也尚不足以把汇总范围说成所有场景固定的开销。

回传的正常/打点内核与主机源码哈希均匹配本地生成器，完整 Search 二进制仍为已验收的 `434ef756…417da`；输入的 index_id 与准备元数据指纹也已归档。A5 SciFact 的 index_id 与本地计划一致，准备报告元数据哈希不同，二者分别保留。原始报告、安装包和逐任务记录未传到本机；正确性/nonce/保护区门槛由 A5 工具执行，本机复核的是回传顺序、汇总数学和指纹。

**决定：保留候选，默认继续使用 1024 原包。** 下一步优先把稀疏收集恢复为独立的专用编译路径，再与本轮候选和原基线比较。当前候选把收集器的 `compact=true` 改为运行时模式判断，这是可隔离验证的代码差异，还不能认定它解释了全部开销。先验证稀疏代价是否缩小，再对 TREC-COVID、MS MARCO 和原千万篇输入做回归；本轮没有这些大库的候选测量，也没有重新测原版 BM25S，不能外推新倍率。

## 改动依据

[容量对照](SELECTOR_CAPACITY.md#v2-a5-六轮结果)已定位 SciFact A 第 8 条查询：4399 个正分候选，超过 4096，通用选择仍约 32553 ticks。扩大容量帮助了 B，但在 8192 稀疏输入带来了中位/P95 代价。本轮保持候选容量 1024，改密集分支的阈值计算。

旧通用分支用三轮 2048/1024 桶 UB 原子直方图求精确阈值。候选在正分总数大于 1024 时，直接对已有的完整局部分数行使用四轮 256 桶 Reg SIMD CDF，随后沿原始文档 ID 流收集严格优于阈值的候选和同分候选。它复用已有的 SIMD 阈值代码，采用相同的 FP32 分数位序和同分 ID 升序规则。

- 不增加候选容量：正分紧凑数组仍为 1024；完整局部行上限仍为 8192。
- 不新增 UB 数组：TBuf 仍为 **50848 字节**，动态 UB 声明仍为 **65536 字节**，沿用原 DCache/编译器预留检查。
- 正分不超过 1024 时沿用原压紧/选择算法；全零或正分不超过 K 时保留原通用处理；非法输入状态不得进入新分支。
- 密集 CDF 输入在寄存器中清除符号位：对已验证的非负有限分数，这只会把 `-0` 规范为 `+0` 的排序键。输入/输出原始 FP32 位不改写。CDF 总数最多 8192，不超过 uint16 容量。
- 新收集模式 3 是内部 UB 控制值，与诊断 `debug=3` 属性无关。对整个原始 ID 行扫描，不把有界数组之外的候选截掉。

评分、追加、Merge、主机 Search、安装接口及安全编译参数沿用原实现。新增 SIMD 代码、分支和寄存器需求仍可能影响编译布局或稀疏性能，所以不能以“UB 没增大”承诺零开销。

本地检查覆盖 1025/2049/4097 非整向量长度、4399 与 8192 密集候选、K1/10/32、FP32 子正规值/相邻值/最大有限值、正负零和大量同分；新阈值与完整排序比较。相同 SciFact 本地准备数据的 16 行严格 oracle 也已对齐，其中 13 行会进入新分支。这些是 CPU 数学检查，不代表设备已运行。

## A5 构建

复用 `capacity_build_02` 中已验收的 1024 正常/打点包与两个诊断主机程序。核对源与安装指纹后，只生成两个新的候选包；不重建基线或 Merge，也不覆盖已有 vendor。

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 -u standalone/dense_selector.py build --baseline-build results/capacity_build_02 --template /test/w50066029/bm25-npu/results/Bm25FusedScoreTopKUbFix1 --output results/dense_selector_build_01
```

成功状态为 `dense_study_packages_built_device_pending`。失败时保留新目录，并回传 `build.json` 和对应 `normal_build.log` 或 `profile_build.log` 的末尾；不继续设备测量。

## 相同输入的六轮对照

在预留 CPU 核窗口执行。下面的 `numactl` 只负责放置，不会预留核。

```bash
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
numactl --physcpubind=0-31 --membind=0 python3 -u standalone/dense_selector.py run --build results/dense_selector_build_01 --host-build results/indexsdk_alloc_host_01/build --output results/dense_selector_a5_01 --device 5 --workload scifact=results/scifact_data_01/performance --workload sparse8192=/test/w50066029/bm25-npu/results/score_bitmap_scales1/inputs/8192 --workload sparse65536=/test/w50066029/bm25-npu/results/score_bitmap_scales1/inputs/65536
```

先对基线与候选的正常/打点包分别执行扩展边界检查，再执行六轮 AB/BA：基线和候选各占三次先运行位置，正常/诊断先后也逐轮交替。每进程 5 个 A/B 预热周期，A/B 各 30 个计时样本。三个输入共 36 个测量单元；基线和候选都重新测量，不拼接历史轮次。

完整 Search 保留独立全分数、各叶和 Merge 层、严格 ID、保护区、索引回读和重复位检查。每个正常/打点包必须位一致，候选与同期基线 Top-K 也必须位一致。阶段使用已验证的 v2 64 字节任务记录和 nonce/任务号校验。阶段仍以原始 ticks 输出，不能当作正常 Search 的可加分项或未经确认的微秒。

完成后回传 `COMPARISON:` 和 `results/dense_selector_a5_01/report.json`。若失败，回传首个失败日志及有则提供的 `.failure.json`。本轮不重复原版 BM25S 建库，不更改其冻结源码或 CPU 主基准。

采用与否以完整 Search 为主：先看 A 第 8 条查询和 A/B 批延迟是否下降，再检查两种稀疏输入的中位/P95 和全部同轮差值。稳定退化如实记为候选代价，不用全机负载笼统排除。本轮已完成，无需重跑现有目录；后续修改使用新的构建和输出目录，保留本轮候选与全部结果。
