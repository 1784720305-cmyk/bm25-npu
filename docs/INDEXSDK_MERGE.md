# 冻结版本合入 IndexSDK 与合入后验收

[集中测试](RELEASE_VALIDATION.md#已完成的集中结果)已完成并冻结 `split`。本阶段只做集成工程与验收，不新增算法、容量或逐数据集参数。8192 稀疏输入约 +4% 的完整 Search 代价继续保留，不因其他输入提升而抹去。

## 源码合入范围

工具从 IndexSDK 的 `59213c4ca5ebb643c031c8b819e90e3a5cc21a8c` 创建独立源码树，不修改参考目录。原始 SDK 文件只改 `feature_retrieval/src/ascendfaiss/ascendhost/CMakeLists.txt`：追加 BM25 模块入口；新增文件放在 SDK 对应的 common、ascendhost、ops、test、ut 层。

- 五个 BM25 实现文件直接编入原 `ascendfaiss_host`，产物仍为 **`libascendfaiss.so`**，不是额外静态 BM25 库旁接。
- 保留 SDK 的安全/优化参数，为 BM25 实现明确补上严格 FP32 参数。`BUILD_ASCEND_BM25=ON` 启用集成，默认关闭时原 SDK 构建行为保持。
- 正常算子源码固定为实测 `d7bd4388…45948`，无性能修改。CANN 生成的 kernel tiling 头从实测安装包捕获，校验 `2393e58e…ab157`，一起放入 SDK 算子源码树，避免隐式依赖模板残留。
- BM25 公开头随 SDK 安装到 `include/faiss/ascend`，依赖的数据头保持目录结构；不复制 CANN 头文件或库。安装后搬迁消费单独检查。
- 新基准调用真实 SDK 共享库里的 **`AscendIndexBm25::Init/Search/Close`**。计时包含公开接口返回的 INT64 labels、FP32 scores，以及接口自身的同步、转换和锁开销。

`prepare` 保存原集中报告和选中包构建报告的原始副本及 SHA-256。后续步骤会复核它们和完整源码树；生成 `indexsdk-bm25.patch` 供评审。Windows 已验证源码补丁可应用原 SDK、CMake 确实把五个实现加入共享库、公开基准依赖该库，以及 C++ 声明语法；这些不代替真实 CANN 编译或设备验收。

## A5：生成合入源码副本

已有数据、包、构建和 `concentrated_preindex_02` 全部保留。下面使用全新输出目录。只复制命令，不复制提示符或历史输出。

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 -u tools/indexsdk_integration.py prepare --sdk-source /test/w50066029/IndexSDK-reference --preindex-report results/concentrated_preindex_02/report.json --selected-build results/dense_release_build_01 --output results/indexsdk_integrated_01
```

产物为 `results/indexsdk_integrated_01/sdk` 和同级 `indexsdk-bm25.patch`。不要把文件逐个覆盖进旧参考目录。

## A5：构建合入后的算子和 SDK

```bash
python3 -u tools/indexsdk_integration.py build --integration results/indexsdk_integrated_01 --selected-build results/dense_release_build_01 --template /test/w50066029/bm25-npu/results/Bm25FusedScoreTopKUbFix1 --output results/indexsdk_merged_build_01 --toolkit-home /usr/local/Ascend/cann-9.2.0/x86_64-linux --faiss-home /usr/local/faiss/faiss1.14.1 --openblas /opt/OpenBLAS/lib/libopenblas.so --clang-tidy /test/w50066029/bm25-tools/bin/clang-tidy --jobs 8
```

此阶段从已合入的正常算子源码重新构建一个 CANN 包，四个有效源码/tiling 头与实测版逐项核对；Merge 复用已验收包。随后构建原 SDK 主机共享库和接口基准，运行七项 CPU 单元检查、真实头文件 clang-tidy、安装和搬迁后的独立消费，并检查实际 SDK 动态库导出 BM25 接口。沿用之前成功构建 SDK 的 CANN/Faiss/OpenBLAS 位置和 `--no-undefined` 链接检查。

成功状态为 `integrated_IndexSDK_build_static_UT_install_passed_device_pending`。失败时保留目录，回传 `build.json` 的 `current_stage` 和对应日志末尾；不要重建原始包或重跑集中矩阵。

### 已编译成功、停在 clang-tidy 的原目录恢复

2026-11-06 的 `indexsdk_merged_build_01` 已通过算子构建安装、SDK 共享库编译及 UT；五个 BM25 实现均通过 clang-tidy。两个 SDK 检查程序失败于未修改的 SDK 依赖头：旧范围规则错误接纳了 `ascendhost/../ascenddaemon` 等路径。现按完整的 BM25 自有头文件名匹配，并保留其通过 `ascendhost/../` 引用时的检查覆盖；检查项和 warnings-as-errors 不变，依赖头的编译错误仍会阻断。

仅对此次停在静态检查、尚未安装的构建，更新代码后执行恢复入口。**不用再次 prepare 或 build**，也不用删除旧目录：

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 -u tools/indexsdk_integration.py resume-build --build results/indexsdk_merged_build_01 --clang-tidy /test/w50066029/bm25-tools/bin/clang-tidy
```

入口先核对合入源码、前序成功阶段、CMake 源码位置及算子包指纹，再将原报告和静态检查日志备份到 `resume_01`。它复用原 SDK 动态库和两个检查程序，执行静态检查、安装、搬迁消费及符号导出检查，并核对三个二进制前后 SHA-256 未变。新日志保存在 `resume_01`，`build.json` 更新为恢复结果；只有成功后才继续下方设备验收。

## A5：通过实际 SDK 接口复验

构建成功后执行：

```bash
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
numactl --physcpubind=0-31 --membind=0 python3 -u tools/indexsdk_integration.py device --build results/indexsdk_merged_build_01 --host-build results/indexsdk_alloc_host_01/build --output results/indexsdk_postmerge_a5_01 --device 5
```

先检查程序实际加载新的 `libascendfaiss.so`，再执行真实 SDK 资源代理的拒绝查询恢复、两次 Init/Close、引用计数与关闭后流仍有效检查。三组真实数据的完整质量包通过公开接口运行，FP32 位、严格排名和 qrels 指标与冻结版逐项比较。

六组固定 Q8 输入各执行六轮 SDK/独立主机 AB/BA，两者使用同一个从合入源码重建的算子包。SDK 返回 INT64 labels，独立主机返回 INT32 IDs，比较时检查整数值和 FP32 分数位；独立主机继续执行全分数、树各层、保护区和索引回读诊断。公开接口基准的循环次序、样本数、输出形状与 SDK 资源生命周期另行校验。

CPU 主基准已在集中阶段完成，保持原版 BM25S 数据记录，本阶段不重复计时 CPU。新报告比较的是 **实际 SDK 公开接口与同期独立主机完整 Search**，不把旧 CPU 数字说成新会话配对结果，也不拿独立主机性能替代 SDK 性能。

成功状态为 `postintegration_actual_public_SDK_quality_and_six_round_Search_checked`，回传 `SDK COMPARISON:` 与 `SDK REPORT:`。检查通过后才做最终集成验收结论；测得的接口开销或回退如实保留。
