# 通用选择路径的 SIMD 阈值候选

当前为隔离实验：本地数学/工具检查通过，**CANN 编译、设备正确性和性能仍待 A5 执行**。已验收的 1024 默认包与源码保持，候选不自动替换现有安装。

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

采用与否以完整 Search 为主：先看 A 第 8 条查询和 A/B 批延迟是否下降，再检查两种稀疏输入的中位/P95 和全部同轮差值。稳定退化如实记为候选代价，不用全机负载笼统排除。候选尚未在 A5 验证，不能提前承诺收益。
