# 第三步：自研A5 Top-K

前提：贡献22条已通过设备0实机核对，累加36组分别在设备0/1通过核对，run1/run2由cmp确认字节一致。这是两个独立进程的跨设备小样本一致性，不代表同进程复用或大规模验收。

最新黄区日志：设备2的6×6/K=3、513×40/K=32及K=1/K=6/重复K=3三项补验都已实际执行并严格核对通过，见[A5Top-K验证记录](A5TopK验证记录.md)。现在进入[同进程三算子串联](同进程串联步骤.md)，本文已有验收不用重复。设备1初始化中断原因仍未确定；BM25S继续作为独立参考与性能对手。

## 先理解输入、处理、输出与验收

输入是NPU累加输出的完整FP32总分，按`[查询数,文章数]`组织；本样本6×6，每行的文章编号为0..5。另一个参数K表示每条查询返回几篇，首轮K=3。

设备给每条查询维护一个按“分数高优先、同分编号小优先”排列的小名单。从文章0开始依次检查；能够进入前K名的文章插入合适位置，后面的名次后移，最后一名挤出。全部文章检查完，再保存完整前K名；不会先按词截断候选。

输出是两个数组：`top_scores[查询数,K]`（FP32）和`top_ids[查询数,K]`（int32）。本轮各18条，共72字节/文件。每行都按排名顺序输出。

验收：NPU调用与同步完成，文章编号逐项严格匹配；选中分数直接复制输入，所以要求FP32字节完全一致，不允许用容差掩盖选错文章。全零查询仍返回K篇，本样本K=3应为编号[0,1,2]，不是空结果；按固定合同保留所有文章参与选择。边界要求1 <= K <= min(文章数,32)，超出时拒绝，不自动缩小K。

同分指实际输入FP32分数相等；不能用浮点容差当同分规则。Python先核对总分与独立公式在容差内对齐，再直接从这份NPU总分排序，避免不同舍入路径影响极近分数的排名。

首轮6条查询的K=3期望编号为：

| 查询编号 | 前3篇文章编号 |
|---|---|
| 0 | [0,2,4] |
| 1 | [4,5,0] |
| 2 | [0,1,2] |
| 3 | [0,1,2] |
| 4 | [4,5,0] |
| 5 | [0,1,2] |

上表依据已收到的累加输出数值；prepare会从用户指定的实际文件重新计算期望。文章4/5同分时编号4先返回，K截断时也按同一规则选择。

## 首次准备输入、生成算子工程（已完成）

更新独立Git源码后，在黄区项目根目录执行以下单行命令。每条成功才继续；已有输出目录会停止，不覆盖。

```bash
cd /test/w50066029/bm25-npu
python3 verify_package.py
python3 standalone/topk_fixture.py prepare --scores results/actual_accumulate_run2.bin --k 3
python3 standalone/create_project.py --operator topk --compute-resource ai_core-ascend950 --compute-unit ascend950
```

输出分别是`results/topk_input`和`results/Bm25TopK`。默认独立vendor为bm25_topk，不覆盖贡献bm25或累加bm25_accumulate。框架采用设备自带msopgen，不依赖IndexSDK或Faiss；生成脚本保持CANN要求的type/value配置结构。

## 首次编译算子（已完成）

```bash
cd /test/w50066029/bm25-npu/results/Bm25TopK
set -o pipefail
bash build.sh 2>&1 | tee ../topk_operator_build.log
```

日志确认`Bm25TopK_ascend950`目标及设备.o生成成功，安装包为`results/Bm25TopK/build_out/custom_opp_openEuler_x86_64.run`，最后一次构建包SHA256为b4ad6bcb9f6a5d7009302fb01388e4a394b2c91964c3bc7fb01e2d0e5d927f5e。源码名称bm25_top_k与当前生成器一致。CANN头文件弃用提示与未使用配置警告未阻止本次构建。

安装已完成，实际生成头文件为`op_api/include/aclnn_bm25_top_k.h`，依赖aclnn/acl_meta.h，函数签名如下。源码包含实际安装头文件，并用static_assert核对签名，不自制替代头文件。

```cpp
aclnnStatus aclnnBm25TopKGetWorkspaceSize(
    const aclTensor *scores, int64_t k,
    const aclTensor *topScoresOut, const aclTensor *topIdsOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnBm25TopK(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

用户提供的`results/topk_dependencies.log`没有not found或undefined symbol，依赖包括同一CANN目录的libascendcl、libnnopbase、libacl_rt、runtime等。此检查没有执行设备，运行时动态加载仍可能出现此前的公共hstu包问题。ATC的libgert缺失接口问题也仍独立保留。

三个vendor的接口库都可能叫libcust_opapi.so，当前分开运行并把当前vendor的op_api/lib放在LD_LIBRARY_PATH最前。完整同进程查询还需处理同名库与设备缓冲串联，本轮不假装已经完成。

## 首次编译调用器（已完成）

更新独立Git代码后，从根目录执行下面的单行命令；每条成功才继续。使用新构建目录，已有贡献/累加程序不重建。

```bash
cd /test/w50066029/bm25-npu
python3 verify_package.py
set -o pipefail
cmake -S npu_prototype/host -B results/aclnn_topk_host_build -DBM25_RUNNER=topk -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 -DBM25_OPP_ROOT=/test/w50066029/bm25-npu/results/custom_opp/vendors/bm25_topk 2>&1 | tee results/aclnn_topk_configure.log
cmake --build results/aclnn_topk_host_build -j 2>&1 | tee results/aclnn_topk_build.log
```

## 首次依赖检查、运行和核对（设备2已通过）

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
source /test/w50066029/bm25-npu/results/custom_opp/vendors/bm25_topk/bin/set_env.bash
export LD_LIBRARY_PATH="/test/w50066029/bm25-npu/results/custom_opp/vendors/bm25_topk/op_api/lib:/usr/local/Ascend/cann-9.2.0/lib64:${LD_LIBRARY_PATH:-}"
ldd -r results/aclnn_topk_host_build/bm25_topk_runner > results/aclnn_topk_runner_dependencies.log 2>&1
cat results/aclnn_topk_runner_dependencies.log
```

确认libcust_opapi来自bm25_topk目录，且没有缺库/未解析符号，才运行。Python入口读取已准备的元信息，检查尺寸与输入指纹，把整数参数交给实际C++调用器；不计算或填充Top-K替代输出。

```bash
python3 standalone/run_topk.py --input results/topk_input --output results/topk_run1 --device 1 2>&1 | tee results/aclnn_topk_run1.log
```

成功后，输出目录包含top_scores.bin、top_ids.bin和run_metadata.json；本样本两个bin文件各72字节。元信息只记录调用器成功和输入/输出指纹，排名仍待核对。继续执行：

```bash
python3 standalone/topk_fixture.py check --directory results/topk_input --actual-scores results/topk_run1/top_scores.bin --actual-ids results/topk_run1/top_ids.bin 2>&1 | tee results/aclnn_topk_check1.log
```

日志中的执行/同步/下载成功、编号和分数字节都通过核对，才确认本次Top-K正确。输出目录已存在时停止；重复运行用topk_run2等新目录，不覆盖。运行失败不输出CPU结果，不生成成功元信息；若磁盘保存失败，调用器尽力移除本次新建的不完整输出文件并退出失败。

C++调用器也可以直接执行：

```bash
results/aclnn_topk_host_build/bm25_topk_runner results/topk_input results/topk_direct_run1 6 6 3 1
```

最后四个参数依次为查询数、文章数、K、设备号。推荐用Python入口，减少手写尺寸错误。C++只读取scores.bin，使用二维连续张量[6,6]和两个[6,3]输出，K为真实接口要求的int64标量。

输出先填NaN分数与-1编号；下载后先检查编号有效、不重复、输出顺序、分数与对应输入字节一致，然后释放资源才保存。这里不在CPU重新选择前K名；完整排名由独立check负责。任何ACL错误、同步失败、非法输入/输出或资源释放失败都停止。

单次时间仅记录提交/同步与输入上传至结果下载，未预热、有阶段打印，不含Python指纹检查、全部初始化、前两阶段及完整查询，不用于加速结论。

## 扩展设备验收（513×40/K=32已通过）

先通过6×6、K=3的真实累加输入，再用同一输入验证K=1和K=6。负向边界K=0、K>文章数、K>32在准备或调用层拒绝；输入NaN、无穷大、负分同样拒绝。

扩大到513条查询×40篇文章的固定合成分数，覆盖全零、全同分、递增、密集同分以及只高一个FP32步长的分数；K=1/3/32都需设备验证。边界样本可这样准备，但准备成功不代表设备验证成功：

```bash
python3 standalone/topk_fixture.py edge --k 32 --output results/topk_edge_k32
```

边界样本也使用run_topk.py调用同一实际程序，再用check指定对应输入与实际输出目录。比如K=32合成输入准备完成且小样本通过后：

```bash
python3 standalone/run_topk.py --input results/topk_edge_k32 --output results/topk_edge_k32_run1 --device 2 2>&1 | tee results/aclnn_topk_edge_k32_run1.log
python3 standalone/topk_fixture.py check --directory results/topk_edge_k32 --actual-scores results/topk_edge_k32_run1/top_scores.bin --actual-ids results/topk_edge_k32_run1/top_ids.bin 2>&1 | tee results/aclnn_topk_edge_k32_check1.log
```

检查器会重新对输入排序，不直接相信expected文件；篡改编号、选中分数或输入会被检测。CPU排序只作小数据正确性标准，不能作为正式性能对手。

## 三项补验脚本（已实机通过）

输入为已准备的results/topk_input（6条查询×6篇文章），以及设备2先前的results/topk_run1。补验不修改这两份材料，会在新目录生成各用例的输入、实际NPU输出、run.log/check.log和总报告report.json。

步骤是K=1（只选一篇）、K=6（按排名返回全部文章）、K=3再运行一次。每项都调用现有C++ NPU程序，独立核对编号和分数字节；最后比较K=3新输出与先前topk_run1，要求两个bin文件分别字节一致。任意失败或手动中断立即停止，保留失败报告，不在CPU算替代输出。

同步新增Python文件及清单后，在当前已加载bm25_topk环境的终端执行：

```bash
cd /test/w50066029/bm25-npu
python3 verify_package.py
set -o pipefail
python3 standalone/validate_topk.py --input results/topk_input --previous results/topk_run1 --output results/topk_validation1 --device 2 2>&1 | tee results/topk_validation1.log
```

成功标准是输出“三项补验全部通过”，且results/topk_validation1/report.json中整体status与三项case.status均为passed。脚本调用子进程会把设备阶段实时打印，同时保存每项run.log；初始已完成的检查不会代替这三次新设备调用。

已有输出目录会停止，不覆盖；重跑需要指定新目录。若停在初始化阶段，用该项run.log定位，不误记为通过。脚本只支持文章数<=32的小样本，以便合法测试K=全部文章；最大K=32、513条查询的另一项扩展验证已有日志证据，不重复强制运行。

这三项运行仍是独立进程，不是同一进程内反复调用或正式性能统计。本机只用明确CPU模拟测试检查了脚本编排、失败停止、报告与日志，不把本机测试记为NPU验证。补验通过后进入同进程的贡献→累加→Top-K设备缓冲串联，当前尚未实现该完整调用路径。

## 关键函数和当前性能限制

- TopKSimt：每个线程负责一条查询，维护最多32个候选，依次插入更优文章；每个输出位置只有一个线程写。异常分数使该查询输出-1，调用器和检查器必须判失败。
- InferShape：根据查询数与K推导两个输出为[查询数,K]；InferDataType指定分数FP32、编号int32。
- Tiling：验证二维输入、K范围和输出大小，再按查询数及A5核数分配任务；数组索引用64位，避免二维乘积在32位溢出。
- reference：CPU完整排序，排序键(-分数,文章编号)，作为与设备插入算法独立的期望标准。
- prepare/write_input：核对实际总分、导出同一FP32输入与元信息；期望数组只给检查使用，不作为设备输出。
- check：校验输入指纹、编号和选中分数字节；不能单凭输出文件证明设备执行，仍需实机日志。
- run_topk.prepare_command：从元信息读取查询数/文章数/K，验证形状、同分规则与输入指纹，准备真实调用命令。
- validate_topk.run_suite：使用同一份FP32总分依次组织三项实际设备调用，每项check通过才继续；更新总体报告，失败立即停止。
- validate_topk.execute/compare_repeat：启动并记录实际C++程序，比较重复输出的原始字节；不计算设备替代输出。
- C++ Resources::Describe：把连续二维数组的形状、步长与类型告诉CANN；分数FP32、编号int32。
- C++ ValidateOutput：只核对下载记录的基本合法性、是否重复、顺序和对应输入分数字节，不在CPU选取替代结果。
- C++ main：初始化→上传总分→执行准备→临时设备内存分配→执行→同步→下载两个数组→释放→保存；不读CPU期望文件。

第一版只并行处理不同查询，每条查询内部顺序扫描全部文章，最坏工作量约为查询数×文章数×K。单条查询无法充分使用所有核，线程私有候选数组也需观察编译后的资源消耗；当前不是最终大数据优化实现。正确性通过后研究分块局部Top-K与合并、设备内串联，依据计时决定，而不是把原型当成性能达标。

当前诊断仍把贡献下载到CPU分组，再上传累加/Top-K；完整查询计时必须包含这些准备与往返传输。后续正式CPU比较必须使用合理优化的同流程计算实现，及独立原版BM25S两类基线。

工程参数与属性声明参考[官方msopgen工具说明](https://www.hiascend.com/document/detail/en/mindstudio/2610/optools/Operatordevelopmenttools/docs/en/user_guide/msopgen_user_guide.md)，命名与Host注册参考[官方命名转换说明](https://asc.gitcode.com/guide/programming_guide/advanced_programming/aclnn_operator_development/appendix/naming_conversion_table.html)和[算子原型定义](https://asc.gitcode.com/guide/programming_guide/advanced_programming/aclnn_operator_development/design_and_implementation/operator_prototype_definition.html)。主机调用仍以黄区当前版本的实际生成头文件为准。
