# IndexSDK 主机接口接入

实际构建、安装消费、共享 SDK 上下文和完整 Search 的两阶段验收命令见 [A5 验收与日志回传](A5_ACCEPTANCE.md)。

当前提供 `faiss::ascend::AscendIndexBm25`，头文件为 `ascendhost/include/index/AscendIndexBm25.h`，可选 CMake 目标为 `bm25_indexsdk`。该接口使用本地 IndexSDK 原始 `common/ErrorCode.h`，没有复制 SDK 错误码或重定义资源代理。它是稀疏词 ID 检索接口，未继承稠密向量 `faiss::Index`。

## 构建

在实际 A5/CANN Linux 环境中：

```bash
cmake -S . -B results/indexsdk_npu_build \
  -DBM25_BUILD_NPU=ON \
  -DBM25_INDEXSDK_SOURCE=/path/to/IndexSDK \
  -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 \
  -DBM25_VENDOR_DIR=/path/to/final/vendor_aliases
cmake --build results/indexsdk_npu_build -j \
  --target bm25_indexsdk bm25_large_search_benchmark bm25_fused_simd_selector_check
```

`BM25_VENDOR_DIR` 保持原基准入口使用的两个别名 `bm25_fused_score_topk` 和 `bm25_topk_merge`。它们应指向实际安装的算子 vendor，设备进程仍需按 BUILD_AND_TEST.md 设置 `ASCEND_CUSTOM_OPP_PATH`。SDK 调用方通过 CMake 链接 `bm25_indexsdk`，并保留其原有 IndexSDK 链接依赖。已提供安装及 CMake 导出规则；Common 组件通过了本机搬迁消费测试，实际 Host/IndexSDK 二进制交付尚待 CANN 环境验证。

只构建接入库时，可增加 `-DBM25_BUILD_BENCHMARKS=OFF -DBUILD_TESTING=OFF`。该配置不要求额外的 CPU OpenMP 依赖。安装及消费示例：

```bash
cmake --install results/indexsdk_npu_build --prefix /path/to/bm25-install --component Development
```

调用方 CMake：

```cmake
find_package(AscendBm25 CONFIG REQUIRED COMPONENTS IndexSDK)
target_link_libraries(my_application PRIVATE AscendBm25::indexsdk)
```

配置调用方时设置 `CMAKE_PREFIX_PATH` 指向安装前缀、`AscendBm25_TOOLKIT_HOME` 指向 CANN、`AscendBm25_INDEXSDK_SOURCE` 指向 IndexSDK 源码。也可直接指定包含 `common/ErrorCode.h` 的 `AscendBm25_INDEXSDK_INCLUDE_DIR`。安装包不复制 CANN 或 IndexSDK 开发头文件。导出目标不保留构建机器的绝对依赖路径，会在消费环境重新定位依赖。

静态检查命令、原始规则兼容性说明和安装验证边界见 [静态检查与安装验证](STATIC_CHECKS.md)。

## 资源契约

1. SDK 调用方完成 ACL/设备/上下文及 `AscendResourcesProxy::initialize()` 的既有初始化，并将活动上下文和该代理的 `shared_ptr` 交给配置。
2. 接口仅借用上下文，保留代理引用，并创建自己的执行流。它不复用 SDK 默认流，也不直接初始化/结束 ACL、设置/重置设备或销毁借用的上下文。
3. 在 Init、Search、Close 和析构前，调用线程必须激活配置中的同一上下文。上下文不符会在提交工作或释放资源前被拒绝，调用方激活正确上下文后可重试。
4. 同一实例的方法使用互斥锁串行执行。Init 包含全局模式的 Graph capture，调用方须协调其他实例及 SDK 活动；本轮没有宣称可与其他全局捕获并发初始化。
5. Close 成功后，调用方才可卸载 SDK 资源、销毁上下文或结束 ACL。上下文错误或清理失败时保留未释放句柄；析构仍无法安全清理时，保留句柄及代理引用至进程退出。

SDK 的资源代理使用计数会影响底层设备释放。保留该引用用于防止代理过早销毁，但调用方仍不得主动调用资源的 uninitialize 或直接销毁借用上下文。

## 调用示例

以下示例中的资源已经由 SDK 初始化；`index` 是已有倒排数组构成的内存对象。索引包含 INT64 offsets、INT32 文档 ID 和 FP32 TF/norm/IDF。接口复制并校验索引，之后调用方修改原对象不会影响检索快照。

```cpp
#include "ascendhost/include/index/AscendIndexBm25.h"

ascend::APP_ERROR SearchBm25(
    const ascend::bm25::Index &index,
    const std::shared_ptr<ascend::AscendResourcesProxy> &resources,
    const std::string &vendors,
    faiss::ascend::Bm25SearchResult &result)
{
    faiss::ascend::Bm25Config config;
    config.deviceId = 0; // 与当前 SDK 设备一致
    config.resources = resources;
    config.vendorDirectory = vendors;
    config.queryCount = 1;
    config.maxQueryWords = 3;
    config.topK = 10;
    if (aclrtGetCurrentContext(&config.context) != ACL_ERROR_NONE) {
        return ascend::APP_ERR_INNER_ERROR;
    }
    faiss::ascend::AscendIndexBm25 model;
    auto code = model.Init(index, config);
    if (code != ascend::APP_ERR_OK) {
        return code;
    }
    code = model.Search({{1, 2, 1}}, result); // 保留重复词与原有词序
    // 需要时调用 model.GetLastError() 记录 aclCode 与 message.data()
    const auto cleanup = model.Close();
    return code == ascend::APP_ERR_OK ? cleanup : code;
}
```

初始化时固定 queryCount、topK 和查询包容量。每条查询最多 64 个词，词 ID 非负，其他数据规模限制沿用完整 Search 的 Plan。结果为展平的 Q×K FP32 scores 和 INT64 labels；排名仍为分数降序、同分文档 ID 升序。

参数错误返回 `APP_ERR_INVALID_PARAM`，未初始化/重复初始化/失败会话返回 `APP_ERR_ILLEGAL_OPERATION`，ACL 与其他内部错误返回 `APP_ERR_INNER_ERROR`。`GetLastError()` 返回最近操作的固定长度消息和原始 ACL 错误码；消息最多保留 1023 字节。成功操作清除旧错误。Search 失败不替换调用方输出；输入错误允许重试，设备执行失败要求 Close 后重新 Init。

## 已验证与待验证

本机通过真实 IndexSDK ErrorCode.h 编译运行错误适配和接口测试。接口测试替换了设备 Session，用于检验状态、索引快照、输出原子性及关闭重试。生命周期测试使用 ACL 替身运行实际 runtime.cpp、fused_runtime.cpp 和 planned_search.cpp，检查所有权及失败清理顺序。替身仅用于 UT，构建 bm25_host/bm25_indexsdk 不引入 mock 目录。

```bash
cmake -S . -B results/indexsdk_cpu_build \
  -DBM25_BUILD_NPU=OFF -DBM25_INDEXSDK_SOURCE=/path/to/IndexSDK
cmake --build results/indexsdk_cpu_build -j
ctest --test-dir results/indexsdk_cpu_build --output-on-failure
```

尚需在实际 CANN 编译数据库上执行 clang-tidy、链接实际 SDK/算子库，并在 A5 进行共享资源运行、完整 Search 正确性和性能回归。当前 Windows UT 和语法检查不覆盖真实 CANN ABI、设备执行或新性能结果。
