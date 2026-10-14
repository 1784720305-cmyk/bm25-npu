# A5自研Top-K实机验证

依据：用户提供的黄区调用器构建、依赖检查、实际设备执行和topk_fixture检查日志。代理没有直接登录黄区或读取设备输出二进制；本记录以用户提供的日志为证据。

## 已通过的阶段与结果

- Top-K算子、配套安装包及aclnn调用器已在CANN 9.2.0环境编译成功，独立vendor为bm25_topk。
- 实际头文件为aclnn_bm25_top_k.h；调用器使用确认后的两阶段接口，不经过ATC/.om，不依赖Faiss/IndexSDK。
- 调用器ldd -r中libcust_opapi.so来自项目bm25_topk目录，未显示缺库或未解析符号。
- 成功执行设备为2。输入是此前NPU累加结果的6×6完整FP32总分，K=3。
- 实际调用顺序：validate_inputs → runtime_init → allocate_and_create_tensors → upload_inputs → get_workspace_size → execute → synchronize → download_output → release_resources → save_output。
- 输出目录为`/test/w50066029/bm25-npu/results/topk_run1`，含top_scores.bin、top_ids.bin（各18项、72字节）以及run_metadata.json。
- 独立检查日志显示6条查询，每条3篇，编号与对应分数完全一致。检查器要求编号逐项匹配、分数字节匹配，不以浮点容差代替排名规则。

| 查询 | 前3篇文章编号 |
|---|---|
| 0 | [0,2,4] |
| 1 | [4,5,0] |
| 2 | [0,1,2] |
| 3 | [0,1,2] |
| 4 | [4,5,0] |
| 5 | [0,1,2] |

结果覆盖多词查询、重复词、空查询、未知词、同分顺序和K截断处的同分选择。本合同保留所有文章参加排名，空/未知词查询全零时返回编号最小的K篇。

黄区证据：results/aclnn_topk_configure.log、results/aclnn_topk_build.log、results/aclnn_topk_runner_dependencies.log、results/aclnn_topk_run1.log、results/aclnn_topk_check1.log及topk_run1实际输出。这些运行产物不纳入源码包清单。

## 初始化中断与环境问题

第一次选择设备1，日志停在stage=runtime_init，随后用户Ctrl-C中断；没有执行/下载/核对证据，不计通过。该阶段含aclInit、设备选择、上下文与流创建等多个调用，现有日志不能确认具体阻塞位置或原因，不能直接断定设备1损坏。随后设备2执行成功不表示该停顿已修复。必要时再细化初始化日志并由环境维护检查设备状态，不擅自重置其他用户设备或结束无关进程。

成功的设备2调用中，公共vendor hstu_v2、hstu_dense_backward仍因系统libstdc++缺少GLIBCXX_3.4.29加载失败。本次BM25 Top-K继续执行且检查通过；不能据此保证这些公共包可用。ATC缺失aclrtHostGetDevicePointerAddrRange的问题仍独立保留。

用户粘贴记录中的LD_LIBRARY_PATH命令含多余转义字符；本次依赖日志显示前两个关键目录仍被正确解析，不能将本次成功作为此拼写可复用的依据。后续以TopK步骤.md中的原始单行命令为准。

## 时间只用于诊断

成功的本次workspace为16,777,216字节（16MiB）；提交/同步0.596285毫秒，输入上传至输出下载313.589毫秒。没有预热/重复统计，包含阶段打印及准备开销，也不含完整初始化、前两阶段、Python分组与完整查询；没有等工作量CPU计时，不能计算加速比。

## 当前里程碑与下一步

三个自研NPU计算阶段均已有小样本实机正确性证据：贡献22条、累加36组（设备0/1一致）、Top-K 18项（设备2）。这三个阶段目前分别启动、通过CPU回传/分组/上传连接，尚未完成同进程设备内串联，也未证明大数据下优于CPU或原版BM25S。

513条查询×40篇文章、K=32扩展验证及K=1/全部文章/重复运行三项补验都已通过，见下节。当前进入[同进程串联](同进程串联步骤.md)，随后扩大贡献/累加边界验证，并使用合理优化的CPU基线进行两类性能比较。

边界数据属于算子正确性测试，不是正式检索语料或性能数据。本记录不宣称Top-K所有规模已验证，也不宣称BM25加速已经达标。

## 扩展验证：513×40，K=32

用户提供后续完整运行与检查日志：设备2完成513条查询×40篇文章、K=32，输出16,416个编号及对应分数，全部严格核对通过。输入由topk_fixture.edge_fixture固定生成，包含全零、全同分、递增、密集同分和仅高一个FP32步长的分数。

结果显示全零/全同分查询返回编号0..31；递增查询返回39..8；仅高一个FP32步长的文章39先返回，随后同分文章按编号排列。检查覆盖全部513条查询，不只是日志打印的前6条。每个输出bin文件为65,664字节，输出目录results/topk_edge_k32_run1，输入目录results/topk_edge_k32。

完整阶段日志包括实际执行、同步、下载、释放和保存。证据为results/aclnn_topk_edge_k32_run1.log、results/aclnn_topk_edge_k32_check1.log、设备输出和run_metadata.json。代理没有读取黄区二进制，以用户日志和工具输出为证据。

本次workspace仍16MiB；提交/同步0.892148毫秒，输入上传至输出下载402.139毫秒，仅作单次诊断，不计算加速比。本次日志未出现公共hstu包报错，不能据此证明此前运行库问题已修复。

这轮验证扩大了查询数并覆盖最大支持K；不代表大量文章的检索性能、K=1/全部文章边界、同进程缓冲复用或完整BM25查询已通过。

## 三项补验已通过：K=1、K=6和重复K=3

用户提供validate_topk.py完整日志，设备2依次完成6×6/K=1、6×6/K=6、6×6/K=3：每项都实际执行、同步、下载并保存，编号和对应分数字节通过独立检查；最后K=3的两个输出与先前topk_run1逐字节一致。总体输出“三项补验全部通过”。报告路径results/topk_validation1/report.json，日志results/topk_validation1.log，各项目录分别为k1、k_all_docs、repeat_original_k，含run.log/check.log及actual设备结果。

每项workspace为16MiB；K=1诊断提交/同步0.528813毫秒、上传至下载356.71毫秒；K=6为0.570816/377.935毫秒；重复K=3为0.544246/362.703毫秒。均不作为性能验收结果。

这补齐了当前小样本的K两端和独立进程重复一致性，不代表同进程重复调用或任意文章数、数据规模的正确性。三算子完整同进程串联仍待验证。
