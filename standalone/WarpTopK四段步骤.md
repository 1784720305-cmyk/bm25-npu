# 已验收warp Top-K接回BM25四段流程

## 当前证据和本轮验收

用户提供的黄区完整日志显示：新独立调用器编译成功，设备2的13组测试全部通过，每组3次同进程调用，编号/分数字节与独立完整排序一致，逐次输出无效值覆写与重复字节检查通过。覆盖65536文章单查询、513查询循环、K=1/32/全部文章、同分和尾批。这个结论限于已测单算子正确性，还没有新四段性能结果。

本轮只改主机接入与检查规则，四个设备算子源码、二进制样本和CPU参考不变。文件清单为99个；无需重新编译或安装算子。新增版本 `document_sharded_warp_sort32_topk_v1`，旧版本 `document_sharded_topk_v1` 和默认parallel选项保留原规则。

输入仍是TF、长度项、IDF及offsets；NPU在同一进程、同一stream依次执行贡献、累加、新局部Top-K、原跨块合并，整链同步一次，中间结果留在设备。输出六数组：贡献、完整文章总分、局部分数/编号、最终分数/编号。验收核对每个文章块候选及全部查询最终排名、首末六数组与重复执行字节，再与CPU直接交叉。公式、原容差及精确同分时编号升序均不改。

## 1. 建立独立新四段运行目录

同步本次代码后执行，上一条成功才继续：

```bash
cd /test/w50066029/bm25-npu
source /usr/local/Ascend/cann-9.2.0/set_env.sh
python3 verify_package.py
python3 standalone/prepare_sharded_runtime.py --topk-implementation warp --warp-validation results/topk_warp_validation1 --output results/sharded_warp_runtime
```

这里只建四个vendor别名，不执行设备：复用贡献、累加和合并，局部Top-K指向 `bm25_topk_warp`。会重新核对13组实际验收报告、输入/输出/日志指纹、完整排序及当前安装源码/接口/库，记录验收报告指纹。缺失、改变或未通过即停止，不覆盖旧运行目录。原验收目录不同可用 `--warp-validation` 指定真实路径，不能改报告内容绕过检查。

## 2. 编译新版四段主机程序

```bash
set -o pipefail
cmake -S npu_prototype/sharded_host -B results/sharded_warp_host_build -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 -DBM25_VENDOR_DIR=/test/w50066029/bm25-npu/results/sharded_warp_runtime/vendors 2>&1 | tee results/sharded_warp_configure1.log
cmake --build results/sharded_warp_host_build -j 2>&1 | tee results/sharded_warp_build1.log
```

主机程序通过真实新头文件检查函数类型，运行时绝对路径加载四个库、核对八个函数来源。包装器明确传入实现类型及预期用户工作区；C++独立重新计算并核对实际接口返回值；Python在运行后再检查原始计时记录与精确工作区，旧调用器不能误记为新版通过。

## 3. 先跑小样本及CPU直接交叉

编译成功后才执行：

```bash
python3 standalone/run_sharded_pipeline.py --input results/pipeline_input --output results/sharded_warp_small_s2_run1 --runner results/sharded_warp_host_build/bm25_sharded_benchmark --vendor-dir results/sharded_warp_runtime/vendors --shards 2 --device 2 --warmup 2 --repeats 3 2>&1 | tee results/sharded_warp_small_s2_run1.log
```

预期22条贡献、36组总分、18个最终编号/分数，局部Top-K用户工作区4608字节。包装器自动核对六数组、局部/最终完整排序及首末一致。任一步失败就停止。用当前已编译CPU参考补同输入直接交叉，这轮只验正确性，小样本耗时不作性能成绩：

```bash
python3 standalone/run_cpu.py --input results/pipeline_input --output results/sharded_warp_small_cpu_t8_run1 --threads 8 --warmup 2 --repeats 3 2>&1 | tee results/sharded_warp_small_cpu_t8_run1.log
python3 standalone/compare_compute_outputs.py --input results/pipeline_input --cpu results/sharded_warp_small_cpu_t8_run1 --npu results/sharded_warp_small_s2_run1 --report results/sharded_warp_small_s2_cross1.json 2>&1 | tee results/sharded_warp_small_s2_cross1.log
```

结果目录已存在时换新名字，保留记录。设备2忙时可选其他空闲950，避开设备4，记录实际设备号。脚本只给子进程设置这四个确定vendor，无需终端source多个同名算子包。

## 4. 小样本通过后才测原65536文章输入

```bash
python3 standalone/run_sharded_pipeline.py --input results/normal_64k_q1 --output results/sharded_warp_64k_q1_s32_run1 --runner results/sharded_warp_host_build/bm25_sharded_benchmark --vendor-dir results/sharded_warp_runtime/vendors --shards 32 --device 2 --warmup 5 --repeats 30 2>&1 | tee results/sharded_warp_64k_q1_s32_run1.log
python3 standalone/compare_compute_outputs.py --input results/normal_64k_q1 --cpu results/normal_64k_q1_case1/cpu_t8 --npu results/sharded_warp_64k_q1_s32_run1 --report results/sharded_warp_64k_q1_s32_cross1.json 2>&1 | tee results/sharded_warp_64k_q1_s32_cross1.log
```

已有CPU目录在这里只作结果核对，历史CPU时间不能当作本轮匹配复测。新用户工作区为49152字节；系统部分以同一平台贡献接口实际返回值作基数，局部Top-K总大小须精确等于基数+用户部分。四次提交/一次同步、每轮新executor、驻留/含传输边界和计时样本规则不变；计时不包括13组历史报告复核、公式/排序检查或文件读写，也不含倒排展开等完整检索成本。

正确后再用同一输入、相同主机程序和计时边界，复测旧/新运行目录及合理CPU线程配置，保留多轮全部样本。旧运行目录可以给同一新编译程序作旧版对照，默认路线不变。`confirm_sharded.py` 与 `analyze_pipeline_profile.py` 已支持新版本及精确工作区检查；随后可复用它们做多轮匹配复测和设备耗时诊断。采集时间不作普通性能成绩，不以某次最低时间认定加速。

## 关键函数与验证范围

- `validate_warp_acceptance`：重新检查已通过单元验收的13组报告、文件与原始分数排序。
- `installed/validate_runtime`：选择明确版本的真实库，检查源码、接口、库指纹和内存公式。
- C++ `iteration`：每轮准备四个执行器，再连续提交四段并同步；不把CPU期望文件传给设备。
- `check_candidates`：以实际设备完整文章总分重新排序，逐块核对候选，再核对最终前K。

本机通过新旧编排、验收失败保护、六数组核对、旧调用器/错误工作区拒绝、新版本复测及合成CSV读表检查，旧三段/四段相关检查也通过。这些用测试替身或合成CSV，不是设备运行。新增四段主机改动仍待黄区编译和设备执行，不能把单算子正确性当成四段或性能达标。

本次commit备注：

```text
feat: 将已验收warp Top-K接入BM25四段ACLNN流程

- 独立运行目录明确选择warp库并复核13组设备验收
- 区分新旧工作区规则，保留八个函数来源和六数组校验
- 扩展性能复测及采集分析版本识别，保留旧路线
- 更新设备进度与LF清单，算子源码及二进制样本不变
```
