# A5 完整验收与日志回传

使用 Git 获取代码时，先按 [Git 同步说明](GIT_SYNC.md) 克隆到新目录；下面的 cd 替换成实际克隆目录，其他验收步骤相同。

请把提供的源码 ZIP 解压到新目录，保留原 `bm25-npu` 的数据、vendor、结果和安装。脚本只创建本项目 `results/` 下的新目录，不生成或安装算子，不修改 CANN/IndexSDK 源码或系统 Python。

## 1. 环境和路径

沿用已有 CANN、IndexSDK 运行环境的加载方式。要求 Linux、Python>=3.9、CMake>=3.17、GCC/Clang/OpenMP、实际 CANN 9.2.0，以及可用的 clang-tidy。工具版本与规则不兼容时会停止并保留错误。

本次 A5 验收先在运行终端加载以下环境，再启动 host/device。`LD_LIBRARY_PATH` 之外还需要 CANN 的 OPP、tiling 等配置；SDK 共享资源检查采用 IndexSDK 自带的在线算子模式，因此不依赖另行部署的 SDK 离线 OM 模型目录。

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MX_INDEX_USE_ONLINEOP=1
```

此次加载后 `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0`，`ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.2.0/opp`。共享检查继续使用实际 SDK 资源代理和 BM25 自定义算子；此模式不代表已经验证 SDK 的离线模型功能。业务应用已有离线模型和资源初始化配置时，应沿用其业务配置。

`SDK_LIBRARY` 必须是实际构建/安装的 IndexSDK 主机库。`SDK_SOURCE` 应与该库对应，并包含 `feature_retrieval/src/ascendfaiss`。`TIDY` 可写绝对路径或 PATH 中的 `clang-tidy`。

```bash
cd /test/w50066029/bm25-npu-indexsdk

# 修改这三个值为 A5 上的实际位置：
SDK_SOURCE="/实际路径/IndexSDK"
SDK_LIBRARY="/实际安装路径/host/lib/libascendfaiss.so"
TIDY="/实际工具路径/clang-tidy"

# 沿用之前的安装和数据，目录有变动时调整：
VENDORS="/test/w50066029/bm25-npu/results/final_runtime/vendors"
INPUTS="/test/w50066029/bm25-npu/results/large_scale_close1/inputs/10000000"
RUN="results/indexsdk_a5_acceptance_03"
```

解压位置不同时调整 cd。host 阶段的 RUN 必须不存在；失败结果也保留，下次 host 使用 `_02` 等新目录。device 阶段使用已通过 host 的同一个 RUN。

## 2. 先运行 host

```bash
python3 verify_package.py
python3 tools/run_a5_acceptance.py \
  --phase host --output "$RUN" \
  --toolkit-home /usr/local/Ascend/cann-9.2.0 \
  --indexsdk-source "$SDK_SOURCE" \
  --sdk-library "$SDK_LIBRARY" \
  --vendors "$VENDORS" \
  --clang-tidy "$TIDY" \
  --jobs 8
```

依次执行包指纹、验收判定单元测试、实际主机/SDK 库及设备检查程序构建、CTest、实际编译数据库上的 clang-tidy，以及 Common/IndexSDK 安装树搬迁后的独立调用方编译和运行。IndexSDK 调用方引用公开接口，检查未初始化/缺少资源时的状态码和输出保持；它会链接并加载实际主机/CANN 依赖，不提交设备查询。

成功状态应为 `host_build_install_static_passed_device_pending`，仅代表 host 阶段通过。

如果报缺少 `securec.h`，在新 RUN 的 host 命令增加 `--securec-include /实际包含securec.h的目录`。不要创建假的头文件或库绕过检查。缺工具、规则不兼容、编译或链接失败时，直接回传生成的日志包，先保留现场。

### 本次 openEuler A5 环境的依赖准备

参考源码固定为 IndexSDK `59213c4ca5ebb643c031c8b819e90e3a5cc21a8c`，位于 `/test/w50066029/IndexSDK-reference`。主机库单独从 `feature_retrieval/src/ascendfaiss/ascendhost` 构建，关闭 `BUILD_ASCEND_SEARCH` 和 `BUILD_OCK`，使用已有 `/usr/local/faiss/faiss1.14.1`、`/opt/OpenBLAS` 及 `/usr/local/Ascend/cann-9.2.0/x86_64-linux`。

在严格的 `-Wl,--no-undefined` 链接检查下，上游主机目标缺少 OpenBLAS 显式链接，会出现 `sgemm_`、`sgeqrf_`、`sorgqr_` 未定义。此次在该独立构建目录配置 `-DCMAKE_CXX_STANDARD_LIBRARIES:STRING=/opt/OpenBLAS/lib/libopenblas.so` 后，用户回传的构建及 `ldd -r` 检查通过。验收进程还需在 `LD_LIBRARY_PATH` 保留 Faiss 和 OpenBLAS 的 `lib` 目录。

BM25 `8b7c910` 首次 host 验收停在 clang-tidy：CANN 间接头文件依赖及独立 clang-tidy 的 GCC `omp.h` 搜索路径。修复提交 `9a99bbc` 已在 `results/indexsdk_a5_acceptance_02` 完成 host 和 device 验收。上面的 `_03` 仅作未来复现的新目录示例；已有成功结果保留即可。

## 3. host 成功后运行 device

INPUTS 必须是既有 large_corpus 目录，包含 `preparation_report.json`。脚本验证并重用该目录，不复制/改写数据。千万篇六轮完整分数诊断会新增二十多 GB 结果；基准入口会检查内存和磁盘预算。

```bash
python3 tools/run_a5_acceptance.py \
  --phase device --output "$RUN" \
  --device 5 --input "$INPUTS"
```

自动使用 host 已记录的 CANN、SDK、vendor 和二进制，执行前后核对 SHA256：

1. 生成 1024 篇的小语料，供共享 SDK 上下文检查使用。
2. 54 次选择检查、36 次分级合并检查。
3. `bm25_sdk_shared_check` 使用真实 `AscendResourcesProxy`、SDK 默认流和当前上下文，执行 A/B 查询并对照独立 oracle；验证非法输入不改输出且可重试、重复 A 的 FP32 位不变；在同一 SDK 资源上执行两次 Init/Close，确认关闭 BM25 后 SDK 上下文、设备和默认流仍有效。
4. 在 INPUTS 上运行既有六轮完整 Search 对照，使用本次新构建的 NPU/CPU 二进制，验证编译来源、结果和性能判定。

device 重试使用新的 `device_01`、`device_02` 子目录。改代码、SDK/CANN 库、vendor API 或二进制后，不能续用旧 host 结果，须新 RUN 重新验证。

## 4. 返回码与日志

| 返回码/状态 | 含义 |
|---|---|
| host 返回 0 | 构建、安装消费、静态检查通过；设备待验收 |
| device 返回 0 / `all_requested_checks_passed` | 本次工作负载的设备正确性及性能目标通过 |
| device 返回 2 / `device_correctness_passed_performance_target_not_met` | 正确性通过，性能目标未通过，保留原始结果分析 |
| 返回 1 / `failed` | 某阶段失败，查看 failed_stage、error 和阶段日志 |

脚本最后打印 REPORT 和 LOG BUNDLE，成功、失败均收集日志，例如：

```text
results/indexsdk_a5_acceptance_01/report.json
results/indexsdk_a5_acceptance_01/acceptance_logs_01.zip
```

device 后 ZIP 序号递增。回传最新 ZIP 和顶层 report.json 即可。ZIP 只收集日志和 JSON 报告，不包含索引、全分数 `.bin`、CANN/SDK 二进制或原安装。完整结果仍留在 RUN 内。

首次可以只回传 host 日志，解决真实环境编译/链接问题后再做设备阶段。当前 Windows 本地仅验证流程判定、入口和声明替身下的 C++ 静态检查，没有运行上述真实 CANN/SDK/A5 流程。

## 5. 2026-11-04 A5 验收记录

验收代码为 `9a99bbc6719bb7144db3ebca8303393496db32bb`，IndexSDK 参考为 `59213c4ca5ebb643c031c8b819e90e3a5cc21a8c`。用户回传报告的包清单、源码锁和 SDK 头文件指纹已与本地对应版本核对；选择、合并、共享资源和六轮性能判定也已根据回传数据重新计算，通过。

- host：真实构建、单元测试、生产 clang-tidy、Common/IndexSDK 安装搬迁消费全部通过。
- device_03：54 次选择、36 次合并、共享 SDK 资源检查、千万篇六轮完整 Search 全部通过，最终状态为 `all_requested_checks_passed`。
- 共享检查覆盖 Q8/K10、两次 Init/Close、A/B oracle、非法输入恢复、重复查询位一致及 BM25 关闭后的 SDK 流可用性。

下表是六个轮次各自统计量的中位数；CPU 每轮取两种 32 线程实现中的较快者。

| 查询集 | NPU 中位耗时 ms | CPU 中位耗时 ms | NPU P95 ms | CPU P95 ms | 本会话倍率 |
|---|---:|---:|---:|---:|---:|
| A | 2.003753 | 12.122969 | 2.009920 | 34.208726 | 6.050 |
| B | 1.979221 | 13.799904 | 2.010105 | 36.964144 | 6.972 |

A/B 均为 6/6 轮领先。CPU 的逐轮中位耗时范围分别为 6.190～18.799 ms 和 6.107～21.331 ms，波动明显；上述倍率仅描述本次固定合成工作负载及会话，不作为结构整改提升算子性能或所有环境稳定领先的结论。

`device_01` 因未加载完整 CANN 环境出现 tiling 注册初始化错误；`device_02` 因 SDK 默认离线模型目录读取失败返回 100014。加载 CANN `set_env.sh` 并设置 `MX_INDEX_USE_ONLINEOP=1` 后，`device_03` 通过。原 report 顶层仍保留旧尝试的 `failed_stage/error` 字段，历史错误没有从报告中删除；最终状态和最后一次尝试的阶段、子报告均为通过。

报告摘要归档为 [indexsdk_a5_acceptance_20261104.json](../evidence/indexsdk_a5_acceptance_20261104.json)。完整日志包仍位于 A5：

```text
/test/w50066029/bm25-npu-indexsdk/results/indexsdk_a5_acceptance_02/report.json
/test/w50066029/bm25-npu-indexsdk/results/indexsdk_a5_acceptance_02/acceptance_logs_04.zip
/test/w50066029/bm25-npu-indexsdk/results/indexsdk_a5_acceptance_02/device_03/full_search/report.json
```

本记录及文档更新没有改变 `9a99bbc` 的生产源码、算子或构建/验收脚本，无需为文档更新重复设备测量。新的源码修改或重新验收仍需新 RUN。
