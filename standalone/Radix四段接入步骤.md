# Radix Top-K接回BM25四段流程

最新状态：用户黄区日志确认，本页四段主机已编译，原6篇文章及正常65536文章/S=32均实机正确，CPU直接交叉通过。下一步执行 [固定条件性能复测](Radix性能复测步骤.md)，本页命令保留供必要回归，不要求重复已通过的运行。

用户提供的黄区日志已确认：设备2的两组smoke、完整19组独立Radix验收全部通过。全套19组每组5次，共95次设备调用；连同smoke共105次。全部检查编号、原FP32分数字节、交替输入和重复输出。独立调用器已编译并实际执行；这仍不是四段或性能结论。

本次复用已有四段主机，新增Radix实现选择、验收记录复核和工作区保护。没有新增或修改算子内核，现有贡献、累加、合并和旧Top-K都保留；不依赖IndexSDK/Faiss运行库。**本次只需编译更新后的四段主机，不重生成或重编已安装算子。**

## 输入、处理、输出和验收

输入复用已有`results/pipeline_input`：22条TF/长度项/IDF、36个查询文章组、6条查询×6篇文章，K=3。本轮先把每条查询的6篇文章分成2份，每份3篇。

1. 自研贡献算子计算每条词项贡献。
2. 自研累加算子生成每篇文章总分。
3. Radix读取每份文章总分，按严格规则选本份K篇。
4. 原自研合并算子换回全局文章编号，得到整条查询的K篇。

中间数据留在设备内存里，四段共用一个stream，最后统一同步。主机只准备输入、提交算子和核对结果。

输出六份实际设备数组：贡献、完整总分、局部分数/局部编号、最终分数/全局编号。除独立BM25公式核对外，还用实际完整FP32总分做完整排序，检查所有局部候选和最终结果；预热和首末六数组还需字节一致。不会把CPU核对答案写成设备输出。

## 本次关键代码

- `standalone/prepare_sharded_runtime.py`：复核19组完整验收报告、调用命令、二进制调用器、输入输出、设备日志及检查器指纹；每份A/B输出重新完整排序核对。只有通过才建立四个vendor别名，`bm25_topk`指向已验收的`bm25_topk_radix`，其余三个指向原安装。新运行目录独立于旧版和Warp。
- `validate_radix_acceptance`：拒绝只通过smoke、缺少用例或文件改变的报告。记录19组实际返回的一致工作区大小，当前用户日志均为16777216字节；这不是把常数写死在调用逻辑里。
- `npu_prototype/sharded_host/sharded_benchmark.cpp`：新增`radix`选择，显式要求零额外GM候选空间及已验收的API工作区。每次准备接口时核对实际返回值，不同则在提交前停止。8512字节片上算法空间由内核管理，不是主机额外申请的GM候选空间。
- `standalone/run_sharded_pipeline.py`：按选定安装目录设置子进程环境，随后复用原六数组与全排序检查。`validate_topk_workspace`让Radix按自身验收值检查，旧版及Warp继续沿用原保护规则。
- `standalone/confirm_sharded.py`：后续独立性能复测也复用同一工作区检查；本轮不运行正式性能复测。

本机30项新接入辅助检查、原四段/Warp和复测工具回归通过。使用的是明确的模拟文件和进程，只验证保护与检查逻辑，不能代表更新后的四段主机已编译或设备已运行。

## 黄区步骤一：新建运行目录、编译四段主机

提交并同步代码后，逐行执行；任何一条失败都先停下。19组设备验收无需重复，程序会读取并复核原报告。

```bash
cd /test/w50066029/bm25-npu
source /usr/local/Ascend/cann-9.2.0/set_env.sh
python3 verify_package.py
set -o pipefail
python3 standalone/prepare_sharded_runtime.py --topk-implementation radix --radix-validation results/radix_topk_validation1 --output results/radix_sharded_runtime
```

只有新运行目录建立成功，再配置、编译新的主机目录：

```bash
cmake -S npu_prototype/sharded_host -B results/radix_sharded_host_build -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 -DBM25_VENDOR_DIR=/test/w50066029/bm25-npu/results/radix_sharded_runtime/vendors 2>&1 | tee results/radix_sharded_configure1.log
cmake --build results/radix_sharded_host_build -j 2>&1 | tee results/radix_sharded_build1.log
```

## 黄区步骤二：原小样本完整串联

复用已经存在的`pipeline_input`，不用再prepare。调用器编译成功后执行：

```bash
python3 standalone/run_sharded_pipeline.py --input results/pipeline_input --output results/radix_sharded_small_s2_run1 --runner results/radix_sharded_host_build/bm25_sharded_benchmark --vendor-dir results/radix_sharded_runtime/vendors --shards 2 --device 2 --warmup 2 --repeats 3 2>&1 | tee results/radix_sharded_small_s2_run1.log
```

验收标准：显示实际加载Radix库、`topk_implementation=radix`、零额外GM候选空间，四段执行与同步完成；六数组、所有局部候选、最终全局排名及重复字节核对通过。程序打印的3轮计时仅随正确性流程保留，不作稳定性能结论。

## 小样本通过后：正常65536文章输入

```bash
python3 standalone/run_sharded_pipeline.py --input results/normal_64k_q1 --output results/radix_sharded_64k_q1_s32_check1 --runner results/radix_sharded_host_build/bm25_sharded_benchmark --vendor-dir results/radix_sharded_runtime/vendors --shards 32 --device 2 --warmup 2 --repeats 3 2>&1 | tee results/radix_sharded_64k_q1_s32_check1.log
```

该运行全部核对通过后，再与已有同输入CPU结果交叉：

```bash
python3 standalone/compare_compute_outputs.py --input results/normal_64k_q1 --cpu results/normal_64k_q1_case1/cpu_t8 --npu results/radix_sharded_64k_q1_s32_check1 --report results/radix_sharded_64k_q1_s32_cross1.json 2>&1 | tee results/radix_sharded_64k_q1_s32_cross1.log
```

这里复用旧CPU输出只为正确性，不能用本次3轮计时和旧CPU时间宣布加速。完成交叉后，才做相同输入、相同计时范围的旧版/Warp/Radix比较和分段定位；正式速度证据还需至少5轮交替独立复测、合理CPU配置以及BM25S独立对照。局部更快也不直接代表整链或完整查询更快。

若目录已存在，请换新输出名，不覆盖旧证据。可换当前空闲的同规格卡，避开设备4。黄区文件留在黄区，仅提供允许分享的摘要。

建议commit备注：`feat: 将已验收Radix Top-K接入BM25四段调用流程`
