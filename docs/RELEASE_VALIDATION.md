# 收敛优化、集中测试与 IndexSDK 合入

2026-11-06 已约定：**本次只做一轮稀疏路径收尾调整，随后集中测试、冻结版本、合入 IndexSDK，再做合入后验收。** 不为每个数据集新增容量或参数，不因零点几%的变化继续扩展优化分支。编译或正确性缺陷需要修复，但不算作继续追逐性能的理由。

密集候选已有 [SciFact 完整 Search 下降 24.43% / 29.37% 的 A5 结果](DENSE_SELECTOR.md)。其稀疏代价也已归档。本次只把收集器拆成两个独立 VF：稀疏收集函数与原 1024 基线逐字一致，密集收集的 `compact=false` 在编译期固定。阈值函数返回已经算出的路由，不增加一次 UB 路由读取或额外屏障。密集 CDF、评分/追加、同分规则、Merge、主机接口、容量 1024 和 64 KiB UB 声明保持。本地一致性和流程检查通过，**A5 CANN 编译及扩展边界检查已通过；完整质量和性能矩阵仍待复跑**。

## 固定的一次集中测试

三个 NPU 版本都重新测量，不拼接历史轮次：

- `baseline`：原已验收的 1024 包。
- `dense_v1`：刚完成 A5 实测的密集阈值候选。
- `split`：本次稀疏/密集收集函数分离的候选。

| 输入 | 篇数 | 全查询质量 | 性能包 | 阶段打点 |
|---|---:|---|---|---|
| SciFact | 5183 | 300 条查询，三版 NPU | A/B 各 Q8，六轮 | 三版均测 |
| TREC-COVID | 171332 | 50 条查询，三版 NPU | A/B 各 Q8，六轮 | 不测 |
| MS MARCO | 8841823 | 43 条 TREC-DL test 查询，三版 NPU | A/B 各 Q8，六轮 | 不测 |
| 原 sparse8192 | 8192 | 完整分数与严格 oracle | A/B 各 Q8，六轮 | 三版均测 |
| 原 sparse65536 | 65536 | 完整分数与严格 oracle | A/B 各 Q8，六轮 | 三版均测 |
| 原 synthetic10m | 10000000 | 完整分数与严格 oracle | A/B 各 Q8，六轮 | 不测 |

每组性能输入在新目录建立一次原版 BM25S 索引，然后测试未修改的原生 `.retrieve`，查询 workers 为 1/8/32；严格适配器不进入 CPU 主基准。原生同分 ID 差异、分数误差与严格 oracle 的一致性单独保留。Q8 时最多只有 8 条查询可同时工作，workers32 不表示 32 条同时查询。完整质量包由三版 NPU 对项目严格 oracle 和 qrels 检查，本轮不额外重复原版 BM25S 的全查询质量测量。

三版 NPU 按六种排列轮换，每版在每个位置各出现两次，任意两版的先后各三次；CPU 配置的前后顺序也平衡。每进程 5 个 A/B 预热周期、A/B 各 30 个计时样本。总计 **108 个 NPU 性能进程、108 个原版 CPU 性能进程、54 个小输入打点进程**，以及 9 个全查询 NPU 质量检查进程。新正常/打点包各执行一次扩展边界检查，已验证的旧包只复核指纹并参加实际 Search，不重复构建。

全查询质量检查独立使用 **1 个 A/B 预热周期和 10 个检查周期**，不进入性能门槛。现有 C++ 程序接受的 `repeats` 范围为 10～100000，Python 在创建输出或启动进程前也检查该范围；性能进程继续使用 30 周期。

完整 Search 保留全分数、所有叶和 Merge 层、严格 ID、索引回读、保护区与重复位检查。三版 NPU 的 Top-K 在各轮、正常/打点及全查询质量间分别做字节一致性检查。阶段仍使用 v2 nonce/64 字节记录，单位为 ticks；不把它换算为未经确认的微秒，也不与正常 Search 耗时相加。

## A5 执行

只复制命令，不复制终端提示符或历史输出。保留已有数据、构建和 vendor。新目录若已存在，换新的后缀，不删除原结果。`numactl` 只指定放置，不能证明 CPU 核已独占预留。

`dense_release_build_01` 的正常/打点包已在 A5 构建并通过边界检查。已有这次构建时，更新代码和完整性检查后直接执行集中测试段，**不要重复执行下面的首次构建命令**。

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 -u standalone/release_validation.py build --baseline-build results/capacity_build_02 --template /test/w50066029/bm25-npu/results/Bm25FusedScoreTopKUbFix1 --output results/dense_release_build_01
```

构建成功后执行一次集中测试：

```bash
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
numactl --physcpubind=0-31 --membind=0 python3 -u standalone/release_validation.py run --build results/dense_release_build_01 --reference-build results/dense_selector_build_01 --host-build results/indexsdk_alloc_host_01/build --dependency-dir /test/w50066029/bm25-npu/results/bm25s_numba_py39 --legacy-root /test/w50066029/bm25-npu --output results/concentrated_preindex_02 --device 5
```

默认复用 `results/{scifact,trec_covid,msmarco}_data_01` 以及旧工程的三组模拟输入，不下载、不重新准备语料。开始前检查所有输入、包/主机指纹、完整输出所需磁盘和主机/设备内存余量；原始诊断输出全部保留。CPU 只在新结果目录建库一次，建库、JIT、初始化和诊断不计入 Search 计时。

成功状态为 `concentrated_preindex_checks_completed_selection_review_pending`，末尾输出 `SUMMARY:` 和 `REPORT:`。回传这两项后统一决定采用哪一版。失败时停止，不自动重试；`report.json` 保存错误和当前阶段/单元，日志保留。不要从头反复运行已完成的矩阵来挑更好的数字。

`concentrated_preindex_01` 在首个 SciFact 基线质量进程因工具传入 `repeats=1` 而失败，日志为 `Invalid integer argument`，未进入性能矩阵。该调用已改为与原质量入口一致的 10 周期，并新增对实际 C++ 参数范围和 Python 调用的回归检查。[修复记录](../evidence/release_validation_cli_fix_20261106.json)保留失败事实；原目录不修改，修复后使用 `_02`，无需重新构建算子或准备语料。

## 冻结与合入后验收

1. 先看所有正确性、全查询质量和指纹检查，再看完整 Search 的六轮中位、P95、配对差值和原版 BM25S 比较；不能只看平均数或 Top-K 阶段。保留已观察到的 4%～5% 阶段代价，只有新实测才能证明它缩小。
2. 有明显实际收益、原有输入无实质性退化后，选择并冻结一个正常算子版本及指纹。本次调整未改善时，评估已验证候选与原包的取舍，不再新增一轮算法/参数候选。正确性失败先修缺陷，不合入有误实现。
3. 在独立 IndexSDK 工作副本完成源码、构建入口和安装规则合入，遵循 `59213c4` 的目录与规范；保留原参考工程。此时才更新默认实现和相应源码锁，实验历史证据继续保留。
4. 使用真实 CANN/IndexSDK 重新构建，执行单元检查、生产 clang-tidy、安装搬迁消费和共享资源生命周期验收。通过 `AscendIndexBm25` 公开接口验证实际集成调用链，并对相同输入复查严格结果、质量和完整 Search；与合入前冻结版比较。最终报告分开列出合入前与合入后，不用独立主机程序的成绩代替 SDK 接口验收。

本次提交只提供收尾候选及集中测试工具，尚未宣称集中测试或 IndexSDK 源码合入完成。
