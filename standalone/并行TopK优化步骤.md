# 优化第一版：每条查询由一组线程共同选Top-K

分段诊断确认Top-K约4.4513ms，贡献约0.0137ms、累加约0.0347ms。已有版本正确，但这份数据中Top-K很慢。具体证据见 [A5分段验证记录.md](A5分段验证记录.md)。

## 2026-10-15实际进展：新版中等规模计算与CPU交叉通过

依据用户提供的黄区日志记录；代理本机未登录服务器，尚未取得原始逐轮JSON。新版已生成、编译、打包并安装到独立vendor `bm25_topk_parallel`。新旧实际生成头文件声明一致，独立运行目录和调用程序编译通过。

设备2的小样本为6查询、6文章、K=3，指定预热5次、正式10轮，整批resident路径中位数0.015283ms。日志确认实际库及六个函数来源，Top-K工作区16782592字节，即系统空间16777216加5376字节候选空间；首次/最终独立检查和重复字节检查通过。

随后设备4运行原 `results/pipeline_medium1`：17查询、4096文章、K=10、368571条贡献、69632组；指定预热5次、正式30轮。Top-K工作区17142784字节，即系统空间加365568字节用户空间。三算子中间主机往返为0，指定预热及末次重复检查通过，首次与最终独立公式/完整排名检查通过。

| 新版设备4、17查询整批resident记录 | 中位数 |
|---|---:|
| 完整计时路径（已准备输入、数据已在设备） | 0.090416ms |
| 每轮API准备 | 0.001972ms |
| 三次提交与末尾同步 | 0.088492ms |

各列分别取中位数，不要求数值精确相加。该路径不含每轮输入传输、CPU输入展开、完整检索；提交与同步时间不是纯内核时间。

设备结果与独立公式的贡献最大误差2.3841858e-07，总分最大误差0.00021362305，在原容差内；Top-K对实际FP32总分的编号/分数字节检查通过。与原4线程C++ CPU输出直接交叉：全部17条查询的170个编号及顺序严格一致，贡献、完整总分、Top-K分数最大差分别为2.384185791015625e-07、4.57763671875e-05、9.5367431640625e-07，均在原容差内。

本次原始黄区产物为 `results/topk_parallel_small1`、`results/topk_parallel_small1.log`、`results/topk_parallel_medium1`、`results/topk_parallel_medium1.log`、`results/topk_parallel_medium1_cross.json`、`results/topk_parallel_medium1_cross.log`；不把运行数据打入源码包。

随后4线程CPU以相同输入、指定预热5次/测量30轮复测，实际线程数4，整批resident中位数 **0.131157ms**。新版设备4为 **0.090416ms**，两次中位数相除约 **1.45倍**；这是当前输入、4线程CPU与新版NPU的驻留计算初测，尚未扫描合理CPU线程数或控制记录系统负载，不代表完整查询、最佳CPU配置或通用硬件结论。新CPU与新版NPU再次直接交叉通过：全部170个编号及顺序严格一致，贡献/总分/Top-K分数最大差与上文一致，未修改容差。

本次CPU产物为 `results/pipeline_medium1_cpu_t4_matched1`、`results/pipeline_medium1_cpu_t4_matched1.log`，直接交叉报告/日志为 `results/topk_parallel_medium1_matched_cross.json` 和 `results/topk_parallel_medium1_matched_cross.log`。CPU此前预热2/10轮的0.169114ms保留为历史，不用于本次比值。

此前旧NPU resident4.52245ms来自设备2，旧设备4的4.50246ms是逐段同步诊断，不与新版直接计算优化倍数。下一步选同一可用设备、相近负载、指定预热5/30轮复测旧版和新版（均普通resident、独立进程选择各自vendor并保留库来源证据）；随后补新版边界、合理CPU线程扫描、含传输、单条/批量及原版BM25S完整查询对比。

### 随后新旧复测：均通过，但设备号未对齐

用户后续附件 `9e86ec86-03bf-4fa1-97e8-dfa85574b3ec` 给出相同调用程序、输入、指定预热5/30轮和普通resident的两次成功运行。日志实际设备为：旧版 **设备1**，新版 **设备4**，不能按输出目录中的matched字样认定为同卡测试。

| 运行目录 | 实际Top-K接口库vendor | 设备 | 整批resident中位数 |
|---|---|---:|---:|
| results/topk_old_medium1_matched1 | bm25_topk | 1 | 4.49012ms |
| results/topk_parallel_medium1_matched1 | bm25_topk_parallel | 4 | 0.087912ms |

六个API函数的库来源检查通过；旧版Top-K工作区16777216字节，新版17142784字节。两者首次/最终公式与完整排名检查通过，指定预热和最后一轮全数组重复检查通过，中间主机往返0。新版API准备中位数0.001542ms，三次提交与末尾同步中位数0.08639ms。新版第二次记录接近前次0.090416ms；不将不同设备旧新版时间相除作为同卡优化结论，也不作为纯Top-K内核加速比。

用户随后说明设备4正在被他人使用，后续避开设备4。当前固定设备1，复用设备1旧版4.49012ms记录，待补新版设备1（建议新目录 `results/topk_parallel_medium1_device1_run1`），再与 `results/pipeline_medium1_cpu_t4_matched1` 直接交叉。此前设备4记录保留为历史，不将其冒充设备1的运行或CPU交叉证据。设备型号无需固定某个卡号，比较时实际设备、输入和计时规则需一致。

## 实现前先理解输入、步骤、输出和验收

输入仍是NPU累加得到的完整文章分数，输出仍为每查询K个文章编号及分数，按分数降序、精确同分按编号升序。贡献、累加、公式和CPU对手不改。

旧版每条查询一个线程扫4096篇文章。实验版根据文章数给每查询32/64/128/256个线程；4096篇时为256个。线程0扫描文章0、256、512……，线程1扫描1、257、513……，每线程最多16篇，分别留下自己的前K个候选。

随后第一组32个线程负责合并候选，每个合并线程管理至多8份有序列表；每次比较列表头，选出全局最好的文章，再推进对应列表。重复K次，得到最终结果。合并也由自研NPU代码完成，未调用CANN TopK或CPU排序。

为什么局部只留K篇够用：一篇文章若排不进自己分组前K，该组已经有K篇按同一“分数/编号”规则优于它，所以它也进不了全局前K。

验收依次为：本机算法模型对独立完整排序；A5编译与安装；实际新旧生成头文件一致；实际库路径/指纹与额外工作区检查；小样本和边界NPU结果；同输入17查询CPU全结果交叉；最后用普通resident模式复测性能。**新版已通过小样本和4096文章设备正确性及CPU交叉；边界补验、统一条件性能比较和完整检索仍待完成。**

## 第一阶段：现在先生成、编译独立实验包

2026-10-15黄区曾发现msopgen入口指定不存在的 `/usr/local/python3.11.0/bin/python3.11`。使用现有Python执行入口帮助后，通过显式 `--msopgen-python python3` 生成工程，后续编译和安装已通过，保留CANN安装文件。重新复现时先核对入口 `gen --help`，成功才显式指定该解释器；不默认它支持全部CANN工具，也不据此认定ATC运行库问题已经修复。

同步代码后校验应为74个文件。每条成功才继续，失败停止并保留日志。新工程输出已存在时换新路径，并在后续命令相应修改。

```bash
cd /test/w50066029/bm25-npu
python3 verify_package.py
set -o pipefail
python3 standalone/create_project.py --operator topk_parallel --compute-resource ai_core-ascend950 --compute-unit ascend950
cd /test/w50066029/bm25-npu/results/Bm25TopKParallel
bash build.sh 2>&1 | tee ../topk_parallel_build1.log
```

使用原来相同IR定义 `Bm25TopK`，函数名和参数不改，但源码来自 `npu_prototype/topk_parallel`，vendor固定为 **bm25_topk_parallel**。原工程、原vendor及旧结果保持可用。已有CANN9.2.0环境即可，沿用已验证生成器路径；这里不调用ATC/OM。

## 第二阶段：编译成功后安装并读取实际接口

安装器文件名沿用已有生成工程。先确认编译成功再执行：

```bash
bash build_out/custom_opp_openEuler_x86_64.run --install-path=/test/w50066029/bm25-npu/results/custom_opp
cd /test/w50066029/bm25-npu
cat results/custom_opp/vendors/bm25_topk_parallel/op_api/include/aclnn_bm25_top_k.h
python3 standalone/prepare_topk_parallel_runtime.py
```

准备工具实际读取旧、新两个生成头文件，核对两阶段函数声明；读取新工程源指纹，检查源码与当前实验版相符，再检查三个已安装API库文件。不猜声明、不写自制头文件。

成功后建立独立 `results/topk_parallel_runtime/vendors`：bm25、bm25_accumulate链接原贡献/累加，bm25_topk链接新bm25_topk_parallel。新旧同名算子不同时放进此次自定义OPP搜索路径。工具打印明确的ASCEND_CUSTOM_OPP_PATH；基准Python入口在本次子进程中自动设置，旧终端全局环境不被改写。

准备工具不会运行设备，生成的元信息状态也是“未设备验证”。库来源核对、工作区保护和设备日志仍是后续验收的一部分。构建/安装日志必须保留，安装器验证共享库失败时停止定位，不跳过检查。

## 第三阶段：新头文件核对后，建立独立主机编译目录

```bash
cmake -S npu_prototype/pipeline_host -B results/topk_parallel_pipeline_host_build -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 -DBM25_VENDOR_DIR=/test/w50066029/bm25-npu/results/topk_parallel_runtime/vendors 2>&1 | tee results/topk_parallel_host_configure1.log
cmake --build results/topk_parallel_pipeline_host_build -j 2>&1 | tee results/topk_parallel_host_build1.log
```

新构建包含新安装头文件，再以已确认函数类型static_assert核对。独立目录避免CMake旧路径缓存，仍按绝对路径加载三个实际API库并核对函数owner。

## 第四阶段：先小样本，再同数据正确性与普通流水线计时

先运行已有6查询/6文章/K=3小输入：

```bash
python3 standalone/run_pipeline_benchmark.py --input results/pipeline_input --output results/topk_parallel_small1 --runner results/topk_parallel_pipeline_host_build/bm25_pipeline_benchmark --vendor-dir results/topk_parallel_runtime/vendors --device 4 --warmup 5 --repeats 10 --mode resident 2>&1 | tee results/topk_parallel_small1.log
```

通过后运行原来同一份4096文章输入：

```bash
python3 standalone/run_pipeline_benchmark.py --input results/pipeline_medium1 --output results/topk_parallel_medium1 --runner results/topk_parallel_pipeline_host_build/bm25_pipeline_benchmark --vendor-dir results/topk_parallel_runtime/vendors --device 4 --warmup 5 --repeats 30 --mode resident 2>&1 | tee results/topk_parallel_medium1.log
python3 standalone/compare_compute_outputs.py --input results/pipeline_medium1 --cpu results/pipeline_medium1_cpu_t4_run1 --npu results/topk_parallel_medium1 --report results/topk_parallel_medium1_cross.json 2>&1 | tee results/topk_parallel_medium1_cross.log
```

Python入口会选择新vendor、核对API库/头文件指纹，向C++传递此次应有的额外工作区大小。C++在首次执行前要求新Top-K工作区包含原系统空间及候选缓冲；如果仍选择旧tiling，会停止，不接受“旧实现照常跑”的结果作为新版验证。

本输入需要365568字节用户候选/标记空间，在系统工作区之外；小样本需要5376字节。分配一次，整个重复运行过程中保留。运行后保存variant标签与输入/输出指纹，后续报告明确新旧版本。

resident仍在三段末尾同步，计时方法与原普通resident一致。旧4.52245ms在设备2、分段在设备4，正式新旧比较应在同一设备4、相近负载和相同预热/重复配置重测旧版，再统一CPU配置；不能只比较不同设备旧时间得出最终结论。

大规模前还要在A5补K=1、K=全部文章、K=32、非整除文章数、同分跨线程、warp尾部和多核循环多查询等检查；本机CPU模型已覆盖相关排序规则，但不替代这些设备检查。全部正确后才扩大语料/批量、含传输和原版BM25S完整查询对比。

## 关键函数与A5适配

- 新 `Tiling`：按文章数选32～256个线程，按查询数和可用Vector核分配块；为候选分数、原始编号和无效值标记计算额外工作区。输入输出校验复用原版。
- `ParallelTopKSimt` 前半段：线程按编号交错读取文章，维护局部候选，写入按“候选名次→线程”排列的GM缓冲，相邻线程访问相邻地址。
- 内存栅栏与块内同步：所有线程写完候选后，先确保写入可见，再等待整组到齐。每条查询只由同一个块合并，不依赖跨核全局屏障。
- 合并后半段：仅完整的第一个32线程组参与，每线程维护至多8个列表头；用warp内交换比较分数和原文章编号，找全局最佳，再推进获胜列表。比较操作不改变所选FP32分数字节。
- `prepare_topk_parallel_runtime`：读取黄区真实新旧接口头文件/库/工程指纹，建立新的运行目录和明确的vendor选择。
- 基准入口：沿用原有首末独立公式、排名、重复字节和CPU交叉验收，失败立即停止，无CPU替代输出。

参考官方A5适用接口：[warp交换](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/Warp_functions/Warp_shfl_functions/asc_shfl_xor.md)、[SIMT同步与内存栅栏](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/sync_and_memory_fence/sync_and_memory_fence_overview.md)、[用户工作区](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/resource_management/workspace/GetUserWorkspace.md)。组内IndexSDK的SIMT入口和workspace组织只作写法参考，无源码链接依赖；实际兼容性以黄区CANN9.2.0编译和设备日志为准。
