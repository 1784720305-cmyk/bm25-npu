# 第三步：自研A5 Top-K

前提：贡献22条已通过设备0实机核对，累加36组分别在设备0/1通过核对，run1/run2由cmp确认字节一致。这是两个独立进程的跨设备小样本一致性，不代表同进程复用或大规模验收。

当前新增自研Top-K Host/Kernel、工程生成入口、输入准备与检查工具；Top-K尚未在A5编译或运行。调用器等黄区实际生成头文件确认后再写，不猜接口签名。BM25S仍作为独立参考与性能对手，不参与设备选取。

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

## 现在先执行：准备输入、生成算子工程

更新独立Git源码后，在黄区项目根目录执行以下单行命令。每条成功才继续；已有输出目录会停止，不覆盖。

```bash
cd /test/w50066029/bm25-npu
python3 verify_package.py
python3 standalone/topk_fixture.py prepare --scores results/actual_accumulate_run2.bin --k 3
python3 standalone/create_project.py --operator topk --compute-resource ai_core-ascend950 --compute-unit ascend950
```

输出分别是`results/topk_input`和`results/Bm25TopK`。默认独立vendor为bm25_topk，不覆盖贡献bm25或累加bm25_accumulate。框架采用设备自带msopgen，不依赖IndexSDK或Faiss；生成脚本保持CANN要求的type/value配置结构。

## 生成成功后：编译算子

```bash
cd /test/w50066029/bm25-npu/results/Bm25TopK
set -o pipefail
bash build.sh 2>&1 | tee ../topk_operator_build.log
```

把完整构建结果发回；确认实际安装包路径、编译问题和源码命名后再安装。命名按官方转换规则将Bm25TopK转换为bm25_top_k；如果当前生成器结果不同，脚本会停止，不覆盖错误位置。

成功安装后需要读取实际生成的aclnn头文件。安装目录预计是`results/custom_opp/vendors/bm25_topk`，以实际安装输出为准。收到真实函数签名和依赖报告后，才补主机调用器与编译配置，再在设备上验证。

三个vendor的接口库都可能叫libcust_opapi.so，当前分开运行并把当前vendor的op_api/lib放在LD_LIBRARY_PATH最前。完整同进程查询还需处理同名库与设备缓冲串联，本轮不假装已经完成。

## 后续设备验收数据（调用器完成后使用）

先通过6×6、K=3的真实累加输入，再用同一输入验证K=1和K=6。负向边界K=0、K>文章数、K>32在准备或调用层拒绝；输入NaN、无穷大、负分同样拒绝。

扩大到513条查询×40篇文章的固定合成分数，覆盖全零、全同分、递增、密集同分以及只高一个FP32步长的分数；K=1/3/32都需设备验证。边界样本可这样准备，但准备成功不代表设备验证成功：

```bash
python3 standalone/topk_fixture.py edge --k 32 --output results/topk_edge_k32
```

调用器执行后，用check核对，实际输出参数需届时填写：

```bash
python3 standalone/topk_fixture.py check --directory results/topk_input --actual-scores results/actual_topk_scores_run1.bin --actual-ids results/actual_topk_ids_run1.bin
```

检查器会重新对输入排序，不直接相信expected文件；篡改编号、选中分数或输入会被检测。CPU排序只作小数据正确性标准，不能作为正式性能对手。

## 关键函数和当前性能限制

- TopKSimt：每个线程负责一条查询，维护最多32个候选，依次插入更优文章；每个输出位置只有一个线程写。异常分数使该查询输出-1，调用器和检查器必须判失败。
- InferShape：根据查询数与K推导两个输出为[查询数,K]；InferDataType指定分数FP32、编号int32。
- Tiling：验证二维输入、K范围和输出大小，再按查询数及A5核数分配任务；数组索引用64位，避免二维乘积在32位溢出。
- reference：CPU完整排序，排序键(-分数,文章编号)，作为与设备插入算法独立的期望标准。
- prepare/write_input：核对实际总分、导出同一FP32输入与元信息；期望数组只给检查使用，不作为设备输出。
- check：校验输入指纹、编号和选中分数字节；不能单凭输出文件证明设备执行，仍需实机日志。

第一版只并行处理不同查询，每条查询内部顺序扫描全部文章，最坏工作量约为查询数×文章数×K。单条查询无法充分使用所有核，线程私有候选数组也需观察编译后的资源消耗；当前不是最终大数据优化实现。正确性通过后研究分块局部Top-K与合并、设备内串联，依据计时决定，而不是把原型当成性能达标。

当前诊断仍把贡献下载到CPU分组，再上传累加/Top-K；完整查询计时必须包含这些准备与往返传输。后续正式CPU比较必须使用合理优化的同流程计算实现，及独立原版BM25S两类基线。

工程参数与属性声明参考[官方msopgen工具说明](https://www.hiascend.com/document/detail/en/mindstudio/2610/optools/Operatordevelopmenttools/docs/en/user_guide/msopgen_user_guide.md)，命名与Host注册参考[官方命名转换说明](https://asc.gitcode.com/guide/programming_guide/advanced_programming/aclnn_operator_development/appendix/naming_conversion_table.html)和[算子原型定义](https://asc.gitcode.com/guide/programming_guide/advanced_programming/aclnn_operator_development/design_and_implementation/operator_prototype_definition.html)。主机调用仍以黄区当前版本的实际生成头文件为准。
