# 官方FP32 Top-K独立对照

**当前暂停执行本路线。** 用户最新要求转回参考IndexSDK后自研，见 [IndexSDK Top-K借鉴路线](IndexSDKTopK借鉴路线.md)。本调用器已在黄区编译/加载并运行；K=32首组通过，K=10/1025长度首组完整排名失败，已知差异涉及正负零边界成员。失败保留，不修改样本或规则，也不继续把下面的性能步骤作为当前任务。以下操作保留为历史参考。

这是参考测量，不替代最终自研算子交付。没有增加新内核，不依赖Faiss、IndexSDK、ATC或OM。黄区文件留黄区，只提供允许分享的终端摘要。

当前依据：黄区实际 `include/aclnnop/aclnn_topk.h` 的函数声明，以及 `lib64/libopapi.so` 实际导出的 `aclnnTopk` 和 `aclnnTopkGetWorkspaceSize` 已由用户提供并核对。先用默认aclnnTopk，不把sortPolicy=1当作严格同分升序保证。

## 实现前先理解输入、步骤、输出

输入直接取已有NPU流水线输出的FP32文章总分，不重新计算BM25。全局模式看作[查询数,文章数]；32分块模式把同一文件看作[查询数×32,文章数/32]，没有改分数、复制输入数组或降精度。32块不是32个真实查询。

主机上传一次，查询临时内存大小，然后执行、同步、下载。每次调用重新准备executor；工作区只在首次分配，后续尺寸变大立即停止。bootstrap和每个检查预热、最后补验都先用无效位模式填输出，再运行并下载核对；正式计时不包含这些填充和下载。

输出为FP32分数和INT64编号。与自研INT32输出的差异应注明。Python对同一份输入完整排序，以“分数降序、同分编号升序”逐项核对首末输出，分数要求原字节一致。没有CPU替代输出；失败不生成通过记录。

## 1. 拉代码后只编译主机程序

以下全部在黄区Linux、项目根目录执行。无需重新生成、编译或安装自研算子。

```bash
cd /test/w50066029/bm25-npu
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -o pipefail
python3 verify_package.py
```

```bash
cmake -S npu_prototype/official_topk_host -B results/official_topk_host_build -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 2>&1 | tee results/official_topk_configure1.log
```

配置成功才编译：

```bash
cmake --build results/official_topk_host_build -j 2>&1 | tee results/official_topk_build1.log
```

## 2. 先加载库，不提交设备任务

```bash
results/official_topk_host_build/bm25_official_topk_runner --check-library /usr/local/Ascend/cann-9.2.0 2>&1 | tee results/official_topk_library1.log
```

应显示两个函数均属于指定libopapi，随后显示未提交设备任务。nm有符号不代表依赖都能加载；若dlopen报告缺失接口或库，停在这里，保留错误。不能因改用aclnn就认定此前运行库缺失接口已修复，也不随意替换系统库。

## 3. 先用现有边界输入做短验收

默认使用之前Warp独立13组验收生成的输入。第一份包含513行×40篇、K=32，第二份包含65行×1025篇、K=10，覆盖全零、全等、递增/递减、重复分数、相邻FP32和正负零。复用输入文件，不调用Warp算子。

```bash
python3 standalone/run_official_topk.py --fixture results/topk_warp_validation1/case4_q513_d40_k32/input --output results/official_topk_edge_k32_run1 --device 2 --warmup 2 --repeats 3 2>&1 | tee results/official_topk_edge_k32_run1.log
```

上一份通过后：

```bash
python3 standalone/run_official_topk.py --fixture results/topk_warp_validation1/case7_q65_d1025_k10/input --output results/official_topk_edge_k10_run1 --device 2 --warmup 2 --repeats 3 2>&1 | tee results/official_topk_edge_k10_run1.log
```

输入目录如不存在，应先找到之前实际13组验收的位置，替换路径；不要为了通过而改原分数或删掉边界行。以上两组是首轮检查，不等于完整13组及所有大小都通过。

## 4. 同一份实际总分：全局与局部分别测

前面的库加载和边界检查均通过后，再使用本次Warp采集产生的实际文章总分。脚本核对原BM25输入、生产者记录、文件指纹及独立公式，然后将这一份实际总分提供给官方Top-K。

全局[1,65536]、K=10：

```bash
python3 standalone/run_official_topk.py --input results/normal_64k_q1 --scores-from results/warp_64k_q1_s32_review1 --output results/official_topk_64k_global_run1 --device 2 --warmup 5 --repeats 30 2>&1 | tee results/official_topk_64k_global_run1.log
```

局部[32,2048]、每块K=10：

```bash
python3 standalone/run_official_topk.py --input results/normal_64k_q1 --scores-from results/warp_64k_q1_s32_review1 --shards 32 --output results/official_topk_64k_local_s32_run1 --device 2 --warmup 5 --repeats 30 2>&1 | tee results/official_topk_64k_local_s32_run1.log
```

两次输入总分的SHA256应相同，只变张量视图。局部输出是每个块内的INT64编号，不是已合并的全局Top-10；该实验没有调用自研最终合并算子。

结果保存于输出目录的device子目录：bootstrap_scores.bin、bootstrap_ids.bin、top_scores.bin、top_ids.bin、official_timing.json；外层保存runner.log与run_metadata.json。输出目录必须全新；失败目录保留，不能被静默覆盖。

## 5. 怎样理解计时与通过记录

正式每轮保存prepare_ms、launch_sync_ms、batch_path_ms。后者包含准备接口、主机检查、一次算子提交和同步；中位数使用nearest-rank 50%，与现有CPU/NPU报告一致。上传、初始化、首次准备/分配、检查下载、输出填充、校验和文件读写均不在正式计时内。

这不是纯设备内核时间、完整BM25查询或加速达标结果。官方单算子的时间不能直接与贡献→累加→局部Top-K→合并四段整链相除。是否确实走AI_VECTOR_CORE以及设备耗时，需后续独立采集确认；现有四段分析器不用于只有官方单算子的采集。

若官方同分行为不满足严格规则，程序停止并保存failed记录。先区分入选集合与次序，不放宽合同；若研究原生行为或增加适配，必须另列配置并计成本。

## 关键函数，给MT解释用

- C++ `Library::Symbol`：按绝对路径加载官方库，并检查函数确实属于该库；--check-library只执行这部分。
- C++ `launch`：按实际头文件调用准备函数，取workspace和新executor，执行并同步；首次后复用已分配的空间。
- C++ `poison/download/Validate`：计时外检查输出确实被写入、编号合法、无重复、分数与输入原字节对应以及同分次序。
- Python `source_input`：核对来源与指纹；全局和分块只改变行列解释，不改实际输入。
- Python `check_output`：用完整排序独立核对K个成员和次序，能发现“分数一样但边界选错文章”的问题。
- Python `validate_timing`：检查轮数、尺寸、重复验收次数、全部计时样本和中位数，不接收不匹配或非有限计时值。

当前状态：黄区已编译并加载官方库，513×40/K=32通过；65×1025/K=10执行完成但严格排名失败，已知首差异涉及正负零边界成员。通用语义与性能未通过。当前暂停此路线，继续自研；此前本地Python3.9及模拟检查不作为设备证据。

建议commit备注：`增加官方FP32 Top-K独立对照调用与严格排名验收`
