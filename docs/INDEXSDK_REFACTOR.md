# IndexSDK 结构整改

基准：本机 `F:/Ascend/IndexSDK`，整改日期 2026-11-03。整改对象是 `bm25-a5`；工作区外层历史原型及原版 BM25S 不属于本次生产源码迁移范围。

## 当前状态（2026-11-04）

本轮目录、构建、接口及资源生命周期整改和约定的 A5 验收已完成。验收代码提交 `9a99bbc` 在真实 CANN 9.2.0 / IndexSDK `59213c4` 环境完成生产静态检查、安装消费、54 次选择、36 次合并、共享 SDK 资源检查和千万篇六轮完整 Search，最终为 `all_requested_checks_passed`。报告摘要已核对源码、SDK 头和二进制关联，并独立复算设备与性能判定，见 [A5 验收记录](A5_ACCEPTANCE.md#5-2026-11-04-a5-验收记录)。

以下各轮记录保留当时的验证范围；其中“待验证”描述不覆盖本段最新进展。规范映射中工具不支持的规则不计作自动检查通过；业务真实语料验收、全部 SDK 功能验证和上游 IndexSDK 仓库合入不属于本次已通过的范围。

## 依据与选择

- 参考 `CONTRIBUTING.md`、`.clang-format`、`.clang-tidy`、`pre-commit/pyproject.toml`。
- 目录参考 `feature_retrieval/src/ascendfaiss/{common,ascendhost,ops,ut,test}`。
- 主机层参考 `ascendhost/include/index/AscendIndexTS.h` 和 `ascendhost/CMakeLists.txt`；算子参考 `ops/ascendc/{op_host,op_kernel}`。
- IndexSDK 根 `.clang-format` 与 pre-commit 中硬编码 Google/Allman 参数不一致。本次采用根文件的 LLVM 派生规则（4 空格、120 列、函数左花括号换行），本项目 hook 使用 `--style=file`，避免两个格式入口相互改写。
- Python 使用 IndexSDK 原始 Ruff 配置；`.clang-tidy` 初期按原文件引入；第五轮已整理为可执行规则集，并保留原始参考与映射记录。

## 已落地

27 个源码/算子描述及许可文件迁到 IndexSDK 对应层，逐文件关系见 `indexsdk_path_mapping.json`。公共层不再依赖原型目录，主机层不依赖测试层；头文件使用防重包含宏和以 ascendfaiss 为根的 include。算子导出名及 CANN 模板构建流程保留。

根 CMake 只承担配置与模块入口，模块、基准程序、UT 各自使用 CMakeLists。保留已有命令行程序名及根构建目录中的程序输出位置，CPU 与 NPU 路径仍可分别构建。引入主机安全编译选项，保留 FP32 顺序要求，Linux 才使用 ELF 链接选项。Common 的 INTERFACE 目标以及 Host/IndexSDK 库均已提供安装和 CMake 导出规则。

C++ 按同一配置格式化，Python 展开密集语句并按 Ruff 配置格式化；原版第三方 BM25S 保持不变。结构检查阻止公共层反向依赖、运行时依赖测试以及恢复旧原型源码路径。

原 `delivery_lock.json` 原样归档为 `evidence/delivery_lock_20261029.json`；当前指纹独立写入 `source_lock.json`。历史设备证据保持原样，新的清单不意味着新的设备验收。

## 验证命令

```bash
python tools/check_structure.py
ruff check --config pre-commit/pyproject.toml standalone tools rebuild_manifest.py verify_package.py
ruff format --check --config pre-commit/pyproject.toml standalone tools rebuild_manifest.py verify_package.py
# 对 feature_retrieval 下全部 .cpp/.h 执行：
clang-format --dry-run --Werror --style=file <source-files>
cmake -S . -B results/indexsdk_cpu_build -DBM25_BUILD_NPU=OFF
cmake --build results/indexsdk_cpu_build -j
ctest --test-dir results/indexsdk_cpu_build --output-on-failure
python standalone/test_final.py
python standalone/test_bm25s_final.py
python verify_package.py
```

第一轮验证：MinGW GCC 15.2/OpenMP 编译三个 CPU 目标成功，CTest 1/1 通过；24 份 C++ 格式检查及结构检查通过；14 份 Python 格式检查和 IndexSDK Ruff 规则检查通过。已有数学模型、语料/独立 oracle 损坏检测和直接 CLI 测试通过；原版 BM25S 本地集成测试通过。未运行 CANN/AscendC 编译或 A5 设备测试。

## 第二轮：主机实现与测试边界

- 增加 `ascendhost/CMakeLists.txt` 和内部静态库 `bm25_host`。资源清理、Graph 资源管理和完整 Search 方法移入三个 `.cpp`；基准程序链接主机库。
- `common/bm25/index_data.h` 提供 `ascend::bm25::Index` 内存输入与校验，不再要求生产索引从目录构造。`validation.h` 独立处理参数和字节数检查，拒绝零元素宽度及乘法溢出。
- 查询解析、二进制文件读写及 `LoadIndex` 移至 `test/ascendhost/bm25/input_io.h`；`SaveDiagnostic` 的定义移至仅由基准程序链接的 `search_diagnostic.cpp`。生产层不再包含文件流操作。
- 删除无调用者的旧 `PlannedNpuSearchSession` 及专用 `Request`、`LoadRequests`、参考比较逻辑。保留最终完整 Search 和独立 CPU 参考实现。
- ACL 资源、算子库和索引辅助类型收拢到 `ascend::bm25`；资源持有类禁止复制和移动，避免隐式复制句柄导致重复释放。尚未统一所有内部命名空间和成员名。
- 编译来源校验覆盖新 `.cpp`、输入辅助头和主机 CMake，不能继续用旧二进制冒充当前源码。

第二轮本机验证：四个 CPU 目标编译成功，CTest 2/2 通过，新增内存索引测试覆盖空输入、重复/乱序/越界倒排、非整数 TF、无效 norm/IDF、错误 offsets 和字节数溢出。六个主机/诊断翻译单元通过临时 ACL 声明下的 C++ 语法检查；该检查不验证真实 CANN ABI、链接或设备行为。

## 第二轮时记录的接入工作

以下是第二轮时的任务清单；后续轮次已完成公开接口、安装、生产静态检查和约定的 A5 验收，最新结论见本文开头。工具不支持的原始规则仍不声称自动通过，上游合入未执行。

1. 文件 I/O 和诊断实现已分离，主要主机方法已拆入 `.cpp`；公开接口已通过 PIMPL 隔离内部 Session，仍需依据实际 clang-tidy 结果统一其余内部命名和可读性问题。
2. 已提供公开 facade、APP_ERROR 适配和借用 SDK 上下文的生命周期模式。已补齐安装与 CMake 导出，Common 组件通过搬迁消费测试；仍需在目标环境验证实际 SDK/CANN 链接、Host/IndexSDK 二进制安装、资源代理行为及共享进程运行，上游合入尚未完成。
3. 在实际 CANN 编译数据库上运行 clang-tidy，处理其余安全、命名和可读性问题。Windows CPU 编译不覆盖 ACL 或 AscendC。
4. A5 上重新构建两个算子和主机程序，运行选择、分级合并、完整 Search 正确性与六轮性能回归，产生当前源码对应的新证据。

这些工作与本次结构整改明确分开记录，历史约 2ms 的设备结果不用于证明新版本性能。

## 第三轮：Session 封装

Session 的 SaveDiagnostic、PoisonForDiagnostic、VerifyIndex 改为私有成员，仅测试层 SessionDiagnostics 友元可调用；破坏性填充与索引回读实现也移入 search_diagnostic.cpp。索引设备指针、计数器和 workspaceBytes 不再公开可写，基准程序改用只读访问方法。诊断入口仍作为私有测试钩子保留，尚未移除全部测试耦合。

本轮沿用当前独立运行的 ACL 所有权，不改变初始化/清理语义。六个主机翻译单元通过临时声明下的语法检查，CPU CTest 2/2 通过；不代表真实 CANN 链接、设备正确性或性能验证。剩余重点仍是 SDK 资源与错误接口适配、公开 facade 和 A5 回归。

## 第四轮：错误接口与共享资源（2026-11-04）

新增 `AscendIndexBm25` 公开接口和可选 `bm25_indexsdk` 静态库，使用 IndexSDK 原始 APP_ERROR 常量。实现 PIMPL、索引快照、实例互斥、INT64 结果 labels、成功后替换输出、固定长度错误详情。参数校验与设备错误分型；一次非法查询不会使 Session 失效，设备执行错误仍要求关闭后重新初始化。

Runtime 增加显式的借用上下文模式：保留 SDK 资源代理引用、创建专用执行流，仅释放自有流，不直接初始化/结束 ACL、设置/重置设备或销毁借用上下文。所有共享模式操作要求调用方先激活绑定上下文。关闭失败保留句柄供重试，析构无法安全清理时同时保留资源代理引用。独立基准程序继续使用原有自主管理路径。

修复 Tensors、Buffers、PlanResources 在释放失败后丢失句柄或过早标记 closed 的问题。清理按捕获结束、同步、销毁 Graph、主机内存、tensor、设备内存、runtime 的顺序推进；仅在释放成功后移除对应句柄。Top-K 公共辅助类型也已收拢到 ascend::bm25 命名空间。

本机 CTest 7/7 通过，包括真实 IndexSDK 错误常量适配、接口状态与输出原子性、实际 runtime.cpp 的所有权测试，以及实际 Graph 资源管理代码的七阶段清理故障注入。ACL 和设备 Session 替身只进入 UT，不能作为真实 ABI、执行结果或性能证据。构建及调用方法见 [IndexSDK 接入说明](INDEXSDK_INTEGRATION.md)。

## 第五轮：可执行静态门禁与开发包（2026-11-04）

原 IndexSDK clang-tidy 配置含字符串内注释和当前工具不识别的规则名，不能直接作为有效门禁。已保留原始参考，整理可执行配置并记录映射及不可用项；不将未知规则记为已通过。clang-tidy 23.1.4 的配置验证通过，修复了多变量声明、直接头文件依赖、using-directive 和特殊成员函数声明等实际问题。

本地 CPU/UT 数据库的 13 个翻译单元、带 ACL 声明替身的完整主机配置的 8 个翻译单元均通过只读检查，两个配置有重复文件。这是本地静态覆盖，不是实际 CANN 工具链或 A5 验收。run_clang_tidy.py 会拒绝缺少生产翻译单元或只有 mock 条目的生产检查请求，并输出源码指纹和替身标记。

新增 Common、Host、IndexSDK 的 CMake 导出和 Development 安装规则。库遵循 host/lib 布局，包配置使用标准 lib/cmake 位置重新定位 CANN/SDK 依赖，不导出工作区绝对路径。Common 安装树搬迁后，独立调用方通过 find_package 编译、运行成功；CPU-only 安装对 Host 请求明确拒绝。Host/IndexSDK 的实际二进制消费仍待目标环境验证。

新增 BM25_BUILD_BENCHMARKS 开关；产品库关闭基准与 UT 时不再要求 CPU OpenMP。将不含 OpenMP 的 Top-K 堆辅助与 CPU 分片函数拆开。评分/合并算子内核保持现有锁定指纹。最终本机 CTest 7/7、结构、格式和 Python lint 检查通过。执行方法见 [静态检查与安装验证](STATIC_CHECKS.md)。

## 第六轮：A5 实际验收入口（实现阶段记录）

按用户选择，提供 run_a5_acceptance.py 的 host/device 两阶段命令及日志回传包。host 使用真实 CANN/SDK 构建、真实生产编译数据库门禁、安装后的 Common/IndexSDK 独立调用方链接与无查询 API 检查；device 执行选择/合并、真实 SDK 资源代理共享上下文检查和已有六轮完整 Search。脚本核对源码、二进制及 CANN/SDK/vendor 指纹，不覆盖原安装或数据。

新增 bm25_sdk_shared_check，需显式指定实际 BM25_INDEXSDK_LIBRARY 和 SDK 资源头所需的 securec.h 目录。它检查 A/B 独立 oracle、非法查询后重试、重复查询位一致、两次 Init/Close，以及 BM25 关闭后 SDK 流/设备/上下文保持有效。安装消费检查增加 IndexSDK 模式，不能用 Common 安装成功替代主机库链接成功。

验收判定的 7 项 Python 单元测试通过；本地 CTest 7/7、Common 安装搬迁和入口检查通过。新 SDK 设备程序只做了声明替身下的本地静态检查，未运行真实 CANN/SDK/A5。Windows 会被实际验收入口拒绝；host 成功状态明确保留 device pending，正确性通过但性能不达标返回 2。完整命令见 [A5 验收与日志回传](A5_ACCEPTANCE.md)。
