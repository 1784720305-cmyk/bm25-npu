# 构建与验收

目标为A5 Linux、Ascend950、CANN9.2.0、GCC/Clang/OpenMP、CMake>=3.17、Python3.9。NPU编排和数据工具只用标准库；可选BM25S使用原私有Numba依赖，不修改系统NumPy或CANN。

## 复用已验收安装

最终vendor仍为 `bm25_fused_score_topk_warp_append_large` 和 `bm25_topk_merge_tree`，没有启用Merge投票。加载此前成功的CANN环境后：

```bash
cd /test/w50066029/bm25-npu
python3 verify_package.py
python3 standalone/build_final.py bind --output results/final_runtime
cmake -S . -B results/final_host_build \
  -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 \
  -DBM25_VENDOR_DIR="$PWD/results/final_runtime/vendors" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build results/final_host_build -j8
ctest --test-dir results/final_host_build --output-on-failure
```

final_runtime须不存在；已经bind过则直接使用，不再次运行bind。新主机构建产物与旧结果分目录保存。根CMake统一构建最终NPU、两CPU和设备检查；旧子CMake已经移除。核心.cpp/.h保留原路径/字节；验证器核对编译记录中的全部对应核心指纹，新构建额外绑定根CMake。历史二进制的额外旧条目只作历史来源，不作为运行依赖。

## 全新算子构建

operators下直接提供源码。已有黄区 `results/Bm25FusedScoreTopKUbFix1` 和 `results/Bm25TopKMerge` 可作实际CANN模板；最终源码替换模板对应三份文件：

```bash
python3 standalone/build_final.py build --output results/final_operator_build
```

默认只构建包不安装。新机器可生成模板后指定：

```bash
mkdir -p results/templates
msopgen gen -i operators/fused/operator.json -f aclnn -c ai_core-ascend950 -lan cpp -out results/templates/fused
msopgen gen -i operators/merge/operator.json -f aclnn -c ai_core-ascend950 -lan cpp -out results/templates/merge
python3 standalone/build_final.py build --output results/final_operator_build \
  --leaf-template results/templates/fused --merge-template results/templates/merge \
  --install
```

实际模板须提供build.sh和CMakePresets.json。若msopgen解释器不可启动，沿用此前验证的Python调用文本入口，不改CANN文件。build在新目录操作，拒绝模板重叠/符号链接；安装时拒绝覆盖已有vendor。新环境模板/编译配置可能不同，源文件相同不代表新二进制已经设备验收。

## 数据与完整Search

固定A/B词ID随包提供，种子20261014、词表1024、每篇16～48词，沿用已测合成规则：

```bash
python3 standalone/large_corpus.py --output results/final_inputs_262k \
  --docs 262144 --vocab 1024 --seed 20261014 \
  --queries-a fixtures/A.txt --queries-b fixtures/B.txt
set -o pipefail
python3 -u standalone/benchmark_final.py \
  --input results/final_inputs_262k --output results/final_search_262k \
  --vendors results/final_runtime/vendors \
  2>&1 | tee results/final_search_262k.console.log
```

也可直接使用原 `results/large_scale_close1/inputs/262144`，不重新造数。规模可选1024/8192/65536/262144/1048576/4194304/10000000，每档独立输出；固定查询数，不以批量扩展替代扩库。外部词ID JSONL通过large_corpus的 `--corpus-jsonl` 输入，仍需业务认可和分布记录。

benchmark六轮正逆角色顺序、5个A/B预热周期、30个正式周期；合理CPU为无目录/目录32/32，小库<=65536增加同二进制1/1。只运行最终NPU，不再运行否决候选。验证完整分数、全部叶/合并层、严格ID、保护区、索引不变和重复输出；诊断在计时外。内存/磁盘预算不足停止，不生成替代设备结果。

可用 `--npu-runner`、`--cpu-runner`、`--directory-runner` 指向既有已验收二进制，避免不必要重建；仍检查保留核心的实际编译指纹。bind只建立两vendor别名，不是重新证明安装产物来源；应使用原验收报告绑定的安装，或按新安装流程检查。

## 可选原版BM25S

复用 `results/bm25s_numba_py39`。新环境用 `python3 standalone/prepare_bm25s_numba.py install --output results/bm25s_numba_py39` 离线安装，保留官方固定wheel和锁文件，不改系统环境。

benchmark命令追加 `--bm25s`，同输入/同会话测原生和严格适配各1/8/32线程；建库/JIT另报。原生同分语义不能冒充严格合同；严格适配若与项目ID不等价，不认定严格等价胜出。已完成BM25S七档无需因清理重跑。

## 新构建的设备检查

新装/新编译需实际设备检查，不用本地模型替代：

```bash
results/final_host_build/bm25_large_search_benchmark --merge-check \
  results/final_runtime/vendors 5 results/final_merge_checks
```

另保留 `bm25_fused_simd_selector_check` 原始54次选择检查，位置参数以源码main的Usage为准，再做小库完整Search诊断。复用旧安装不因源码整理重跑全部旧边界。退出0表示完整测量/正确性通过，是否领先看独立decision，不代表所有规模都赢。

输出必须新目录；失败保留首错和日志。不要删除黄区原结果、模板、依赖或安装。
