# 构建与验收

当前为 IndexSDK 结构整改版；历史实测不等于本版验收。运行设备步骤前，先按 [整改说明](INDEXSDK_REFACTOR.md) 执行格式、结构与 CPU 检查。

**当前整改版请优先使用 [A5 完整验收与日志回传](A5_ACCEPTANCE.md)**。下文保留历史独立运行入口及既有安装用法，历史成功记录不代表当前版本通过。

目标为A5 Linux、Ascend950、CANN9.2.0、GCC/Clang/OpenMP、CMake>=3.17、Python3.9。NPU编排和数据工具只用标准库；可选BM25S使用原私有Numba依赖，不修改系统NumPy或CANN。

已有最终安装的机器只执行“复用已验收安装”分支，然后进行设备检查/完整Search。下方两种模板构建方式是可选替代方案，不要顺序全部执行，不需要为源码整理覆盖安装。

本次黄区日志已确认：final_runtime绑定、5个主机程序编译、CPU ctest 1/1、基于既有模板的两套算子包构建均成功。results/final_operator_build已存在是这次成功构建的结果，应保留，不删除重建。当前msopgen shebang指向缺失的Python3.11，只影响新模板生成分支，不阻塞既有安装验证。

2026-10-29后续验收已完成：补齐ASCEND_CUSTOM_OPP_PATH后，54次选择检查、合并检查和results/final_delivery_10m1千万篇六轮完整Search全部成功，A/B各6/6领先合理CPU且P95更低。下方命令用于复现，现有成功结果目录无需重跑或覆盖。

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

final_runtime须不存在；已经bind过则直接使用，不再次运行bind。新主机构建产物与旧结果分目录保存。当前源码已迁入IndexSDK分层目录并拆分实现，不能复用旧二进制声称对应当前源码；验证器核对当前编译记录中的核心指纹。历史二进制及旧条目只作历史来源。

## 可选：基于既有模板构建包

当前算子源码位于feature_retrieval/src/ascendfaiss/ops/ascendc/bm25/。已有黄区 `results/Bm25FusedScoreTopKUbFix1` 和 `results/Bm25TopKMerge` 可作实际CANN模板；最终源码替换模板对应三份文件：

```bash
python3 standalone/build_final.py build --output results/final_operator_build
```

默认只构建包不安装。已构建成功则停止此分支；现有安装不执行--install。下面仅适用于没有模板和最终vendor的新机器，并要求msopgen入口已能运行，不能与上面共用同一个已存在输出目录。

## 可选替代：新机器生成模板并安装

```bash
mkdir -p results/templates
msopgen gen -i feature_retrieval/src/ascendfaiss/ops/ascendc/bm25/fused/operator.json -f aclnn -c ai_core-ascend950 -lan cpp -out results/templates/fused
msopgen gen -i feature_retrieval/src/ascendfaiss/ops/ascendc/bm25/merge/operator.json -f aclnn -c ai_core-ascend950 -lan cpp -out results/templates/merge
python3 standalone/build_final.py build --output results/final_operator_build \
  --leaf-template results/templates/fused --merge-template results/templates/merge \
  --install
```

实际模板须提供build.sh和CMakePresets.json。若msopgen解释器不可启动，沿用此前验证的Python调用文本入口，不改CANN文件。build在新目录操作，拒绝模板重叠/符号链接；安装时拒绝覆盖已有vendor。新环境模板/编译配置可能不同，源文件相同不代表新二进制已经设备验收。

已绑定旧安装且已编译主机的当前机器，应跳过整个新机器分支。不要为修复bad interpreter擅改msopgen首行、创建系统Python软链接或重装CANN。

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
cd /test/w50066029/bm25-npu
export ASCEND_CUSTOM_OPP_PATH="$(readlink -f results/final_runtime/vendors/bm25_fused_score_topk):$(readlink -f results/final_runtime/vendors/bm25_topk_merge)"
results/final_host_build/bm25_fused_simd_selector_check \
  results/final_runtime/vendors 5 results/final_selector_checks &&
results/final_host_build/bm25_large_search_benchmark --merge-check \
  results/final_runtime/vendors 5 results/final_merge_checks
```

上述环境变量必须在启动两个原生检查程序之前设置。vendors参数用于加载API库，不能替代运行时的自定义算子搜索路径；bind也只创建别名，不会修改父Shell环境。benchmark_final会自行通过final_support.environment向NPU子进程传入相同的两个真实vendor路径。

若已加载Warp Append API，但准备算子时报161001/EZ1009，应先补齐此环境后重试；该信息同时列出配置/二进制/安装等可能原因，不能据此认定缺少AddConfig。保留的fused/merge源码均包含AddConfig("ascend950")。若补齐环境仍失败，保留日志并检查这两个实际vendor目录的配置/二进制，不直接覆盖安装。

两项分别为54次选择与36次分级合并检查，用来确认清理后的新主机入口与既有安装可用；无需重跑旧候选。当前NPU主机只引用不含OpenMP的堆辅助；两CPU目标单独链接OpenMP。

检查通过后，可复用已存在的千万篇数据，验收最终交付编排，无需重建数据、生成新模板或安装算子：

```bash
set -o pipefail
python3 -u standalone/benchmark_final.py \
  --input results/large_scale_close1/inputs/10000000 \
  --output results/final_delivery_10m1 \
  --vendors results/final_runtime/vendors \
  2>&1 | tee results/final_delivery_10m1.console.log
```

退出0表示完整测量/正确性通过，是否领先看独立decision，不代表所有规模都赢。该过程含6轮3角色、18个性能进程/1080次正式Search；不重跑已完成BM25S，也不把新主机编译成功当作性能通过。

输出必须新目录；失败保留首错和日志。不要删除黄区原结果、模板、依赖或安装。
