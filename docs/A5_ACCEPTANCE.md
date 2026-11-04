# A5 完整验收与日志回传

使用 Git 获取代码时，先按 [Git 同步说明](GIT_SYNC.md) 克隆到新目录；下面的 cd 替换成实际克隆目录，其他验收步骤相同。

请把提供的源码 ZIP 解压到新目录，保留原 `bm25-npu` 的数据、vendor、结果和安装。脚本只创建本项目 `results/` 下的新目录，不生成或安装算子，不修改 CANN/IndexSDK 源码或系统 Python。

## 1. 环境和路径

沿用已有 CANN、IndexSDK 运行环境的加载方式。要求 Linux、Python>=3.9、CMake>=3.17、GCC/Clang/OpenMP、实际 CANN 9.2.0，以及可用的 clang-tidy。工具版本与规则不兼容时会停止并保留错误。

`SDK_LIBRARY` 必须是实际构建/安装的 IndexSDK 主机库。`SDK_SOURCE` 应与该库对应，并包含 `feature_retrieval/src/ascendfaiss`。`TIDY` 可写绝对路径或 PATH 中的 `clang-tidy`。

```bash
cd /test/w50066029/bm25-indexsdk-acceptance-20261104

# 修改这三个值为 A5 上的实际位置：
SDK_SOURCE="/实际路径/IndexSDK"
SDK_LIBRARY="/实际安装路径/host/lib/libascendfaiss.so"
TIDY="/实际工具路径/clang-tidy"

# 沿用之前的安装和数据，目录有变动时调整：
VENDORS="/test/w50066029/bm25-npu/results/final_runtime/vendors"
INPUTS="/test/w50066029/bm25-npu/results/large_scale_close1/inputs/10000000"
RUN="results/indexsdk_a5_acceptance_01"
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
