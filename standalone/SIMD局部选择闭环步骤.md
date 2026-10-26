# 本次只决定一个候选的去留

保留原256线程融合版。候选仍用SIMT计算BM25，只把适合的小候选集合交给950的Reg向量指令选择K篇；最后使用已有合并算子。不开新一轮线程、分块或CPU参数扫描。

输入是提前常驻的整库原始TF索引和每次不同的查询词编号。输出仍是分数降序、同分文章编号升序的FP32 Top-K。评分公式、重复查询词的累加顺序、32分块、CPU32线程/32文档分区和完整Search计时边界均保持原样。

## 代码怎样处理

1. `FusedScorePackSimt`：沿用已验收评分代码，得到当前分块的片上分数；按原文章编号保留正分文章，记录各线程有序的来源范围。正分不超过1024且多于K时进入向量路径。
2. `HistogramCdf`：每次处理FP32排序键的8位，用950基础`Histograms(...ACCUMULATE)`指令得到256个累计计数，最多四遍。这里没有把浮点数转成BF16，没有调用官方TopKV2或高层TopK库。
3. `FindSimdThreshold`：仿照官方阈值查找的原则，在累计计数中二分查找K所在桶，逐字节确定完整32位阈值。每遍最多8次标量比较；不额外计算前缀和、不排序全部文章。
4. `FusedCollectSimt`：复用原来的严格入选、真实文章编号同分合并和最终K排序。全零、正分不够K或超过1024时走原设备选择器，不截断、不返回CPU替代结果。

任务循环移到外层AIV包装，评分/整理、向量筛选、最终收集之间增加了VF切换和必要同步。这些不是免费操作，全部计入Search。显式片上缓冲仍为50848字节，动态UB声明仍为64 KiB；只是复用原直方图区域保存256项CDF和来源范围。编译后资源占用另存，缺失字段保持UNKNOWN，不由栈大小推断寄存器溢出。

实现入口：`standalone/fused_simd_sources.py`严格从原256线程源码生成独立副本，评分主体、原选择器和同分收集代码逐段检查；新增基础向量逻辑在`npu_prototype/fused_simd_select/simd_selection.h`。实际生成的完整内核会放在新结果目录的`project/op_kernel/bm25_fused_score_top_k.cpp`，可以直接查看。

借鉴依据：[IndexSDK的SIMT计算/片上选择组合](https://gitcode.com/whxxkkk/IndexSDK/blob/59213c4ca5ebb643c031c8b819e90e3a5cc21a8c/feature_retrieval/src/ascendfaiss/ops/ascendc/op_kernel/ascendc_ivfpq_search_distance_topk_simt.h)；[固定版本官方B32累计直方图](https://gitcode.com/cann/ops-math/blob/a8d06fdaec5a507f9c11b1c8b886b6a0e947aadb/math/top_k_v2/op_kernel/arch35/radix_sort_topk_b32.h)和同目录`radix_sort_topk_b8.h`的寄存器展开方式、`radix_sort_top_k.h`的边界二分原则。对应派生代码保留华为版权和CANN许可，完整许可在`npu_prototype/fused_simd_select/LICENSE-CANN`。不引入这些仓的运行依赖。

[官方Histograms说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/histogram_compute/Histograms.md)确认ACCUMULATE是累计统计、输出uint16，每次只处理一个128桶半区；这里输入最多1024，不会达到65535上限。[9.2.0-beta.2编译器说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/asc-devkit/docs/zh/guide/programming_guide/compilation_and_execution/operator_compilation/bisheng_compiler.md)给出`AscendC::Reg`调用方式。公开文档不是黄区实际头文件的替代，运行脚本会先读取实际声明，编译器再检查完整重载。

## 黄区一次按序运行

先在本地提交并推送本次更新，黄区执行下面整段。不要在已有结果目录上重跑，也不要提前手工安装同名候选vendor。

```bash
cd /test/w50066029/bm25-npu &&
git pull --ff-only &&
source /usr/local/Ascend/cann-9.2.0/set_env.sh &&
python3 verify_package.py &&
set -o pipefail &&
python3 -u -B standalone/fused_simd_select.py --output results/fused_simd_close1 --device 5 2>&1 | tee results/fused_simd_close1.log
```

脚本使用已经通过的原256线程安装、验收报告、采集报告和原Search/CPU调用器，只编译独立候选及受影响的选择器检查程序。候选vendor为`bm25_fused_score_topk_simd`，不会替换原版。

按顺序自动完成：

1. 核对旧版证据和源码；读取实际Reg基础接口并保存声明/指纹；编译独立副本、保存编译日志和可识别的VF资源；安装后读取**实际生成的两阶段接口声明**，与既有调用器逐项匹配。不一致立即停止。
2. 只验收改变的选择器：直接输入原始FP32分数，覆盖正分0、K附近、63/64/65、1023/1024/1025、容量溢出、尾部长度、相邻浮点数、极小值、正负零、同分与真实编号。K=1/10/32，A/B/A输入位置变化，输出每次poison并检查尾部保护区。`debug=2`仅用于这个隔离检查，公开Search仍只用debug0/1。
3. 验收新组合在当前256K文章/Q8/K10/S32上的A/B/A/空/B评分、局部候选及最终结果，核对完整公式、编号、分数字节和索引不变。任一步失败均停止，不继续测速。
4. 正确后原256版/候选版六轮交替独立进程，预热5次、正式30次，报告A/B每轮中位数及p95。计时从CPU已有原始词编号开始，包含整理、上传、设备执行、同步、Top-K下载和既有计时规则内的释放；不把调试全分数下载放入正式计时。索引准备/一次上传不计入单次Search，两边规则相同。
5. A、B都满足六轮全快、中位数改善至少5%、各轮p95的中位数不变差，才保留候选并继续同期CPU32/32六轮。否则结论为保留原256版，不追加变体。

这是当前固定工作负载的操作判据，不是普遍加速证明。即使超过原NPU版本，也必须另看CPU结果；保留历史CPU约0.074 ms这一较快证据，不能因为同期机器负载不同而删除它。Q8结果是整批查询时间，不能当单查询延迟。

## 执行后发什么

允许分享的终端末尾会给出A/B每轮中位数、p95及保留/放弃结论，直接贴这部分即可，不需要导出黄区文件。报告保存在`results/fused_simd_close1/report.json`和`comparison.json`。

若失败，只贴`FAILED`首错及相应日志末尾；不要跳过正确性继续测速。编译错误查看该结果目录`resource_build.log`，选择器错误查看`selector_device.log`。已存在的失败结果保留作依据，不删除后假装首次运行。

本地已完成的是源码及非设备模型/流程检查。CANN编译成功、设备正确、完整Search稳定改善和超过CPU，是后续不同的验收结论。
