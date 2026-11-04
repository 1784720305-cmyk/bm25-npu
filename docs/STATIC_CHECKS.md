# 静态检查与安装验证

## 规则来源与兼容性

IndexSDK 原始 `.clang-tidy` 已保留到 `pre-commit/indexsdk-clang-tidy.reference.yaml`。其 `Checks: |` 字符串内的注释会被当作规则名，且包含多个 clang-tidy 23.1.4 不识别的名称。根 `.clang-tidy` 是其可执行整理版，映射与不可用项记录在 `pre-commit/clang-tidy-mapping.json`。

保留原配置中可执行的规则和关闭项，修正 include-cleaner、unused-parameters、move-const-arg 等名称。原配置对头文件 namespace using 有相互冲突的启用/关闭描述；整理版执行禁止 using-directive 的要求，代码已改用显式 using-declaration。原配置中的无效选项已去除；原始命名选项随参考文件保留，原配置未启用的命名检查不计为自动验收。未声称不存在的规则已通过；不可用名称仍需人工审查或目标工具链支持。

本地验证版本为 clang-tidy 23.1.4。更换工具版本时先验证配置；脚本发现未知规则或配置错误会停止。实际 CANN 编译数据库必须在目标环境复查。

## 目标环境门禁

先按 INDEXSDK_INTEGRATION.md 配置并成功编译实际 NPU/SDK 目标，再运行：

```bash
python tools/run_clang_tidy.py \
  --build-dir results/indexsdk_npu_build \
  --clang-tidy /path/to/clang-tidy
```

默认检查主机实现；缺少生产翻译单元、数据库只有 mock 编译条目、配置不兼容或检查失败都会返回非零。脚本优先选择真实主机条目，避免同一 runtime.cpp 的 UT 编译记录遮蔽实际 CANN 编译记录。`--include-tests` 可纳入基准与 UT，`--fix` 显式启用修复。自动修复的头文件范围限制在本项目公共层和主机层，避免改写外部 IndexSDK 头文件。

日志和 `report.json` 位于构建目录的 `bm25_clang_tidy/`。每项记录源文件 SHA256、返回码和是否使用替身。任何静态检查结果都不代表设备执行验收。

## 本机覆盖

Windows 的本地检查使用 `--allow-test-stubs`，报告明确保留该标记。CPU/UT 编译数据库检查 13 个翻译单元；包含完整主机实现的本地声明数据库另检查 8 个翻译单元。后者仅提供 ACL 声明，不能验证 CANN ABI 或设备行为。两个范围有重复文件，不能相加当作 21 个独立源文件。

本机 MinGW 需要给 clang-tidy 指定目标和 GCC OpenMP 头目录。路径按实际工具安装位置调整：

```powershell
python tools/run_clang_tidy.py `
  --build-dir results/indexsdk_cpu_build `
  --clang-tidy C:/tools/clang-tidy.exe `
  --include-tests --allow-test-stubs `
  --extra-arg-before=--target=x86_64-w64-windows-gnu `
  --extra-arg=-idirafterC:/tools/mingw/lib/gcc/x86_64-w64-mingw32/15.2.0/include
```

OpenMP 头路径放在 `-idirafter`，避免用 GCC 的 stddef.h 覆盖 clang 的内建头。这里不使用伪造的 OpenMP 头。

## 安装及搬迁消费验证

安装规则提供 `Common`、`Host`、`IndexSDK` 三个组件。库安装到 `host/lib`，CMake 包位于标准 `lib/cmake/AscendBm25`，头文件根为 `include/ascendfaiss`。组件随实际构建目标提供；CPU-only 安装只包含 Common，不能被识别成已安装主机库。

```bash
python tools/check_installed_package.py --build-dir results/indexsdk_cpu_build
```

该脚本安装 Development 组件，将安装树复制到另一前缀，检查导出文件不包含工作区绝对路径或测试文件，再通过 `find_package(AscendBm25 COMPONENTS Common)` 构建、运行独立调用方。它还检查 CPU-only 安装对 Host 请求的明确拒绝。产物与日志保留在 `results/installed_package_*/`。

此搬迁测试已在本机验证 Common 组件。Host/IndexSDK 二进制的构建、安装、链接和运行仍须由真实 CANN/SDK 环境验证。
