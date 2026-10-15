# 单查询跨文章块Top-K：首轮实验

## 为什么现在改这一处

用户在设备1实测正常短查询：65536篇文章、1条3词查询、K=10、6034条贡献、65536组，输入为 `results/normal_64k_q1`。实际CPU线程扫描和NPU结果均检查通过，全部10个编号及顺序一致，贡献最大CPU/NPU差4.76837158203125e-07，总分和Top-K分数最大差9.5367431640625e-07，均在原容差内。

| 同一已准备输入的驻留计算 | 中位数 |
|---|---:|
| CPU 1线程 | 0.115343ms |
| CPU 2线程 | 0.08005ms |
| CPU 4线程 | 0.056144ms |
| CPU 8线程 | 0.041542ms |
| CPU 16线程 | 0.051207ms |
| 当前NPU，设备1 | 0.474331ms |

本次最快已测CPU为8线程，NPU约慢11.4倍。该结论是本次单条短查询工作负载，不能拿此前4096文章/17查询相对4线程约1.5倍的优势替代。CPU更多线程并不总是更快，亦未宣称当前CPU实现是所有可能实现中的最快。

原始证据为用户附件185e10ee-ad41-4b04-913a-ceebcb014822，黄区结果目录 `results/normal_64k_q1_case1` 和其中 `report.json`。本机没有运行CPU C++或NPU。65536文章的逐段设备时间尚未采集，不能把全部0.474331ms精确归因给Top-K；但当前源码每条查询只分一个块的限制已确认，且前一工作负载的正式设备采集显示Top-K仍为大头，因此先做这一结构实验。

## 实现前先理解输入、步骤、输出

输入仍为NPU累加得到的完整文章分数。首轮把文章均匀分成S块：65536文章可选S=8、16、32，S=32时每块2048文章。

1. 将原 `[查询数,文章数]` 分数缓冲，描述成 `[查询数*S,每块文章数]`。数据原本连续，计划只新增张量描述，不由CPU搬动或重新排序完整分数。
2. 调用现有自研 `Bm25TopK`，每个文章块独立选K个候选。单查询S=32时可形成32个工作块，替代原来的1个；实际块数仍受设备可用资源约束。候选分数/块内编号直接留在设备。
3. 调用新自研 `Bm25TopKMerge`，读取每块有序的K候选，换算原文章编号，按分数降序、精确同分按原编号升序选最终K。

输出仍为每查询K个原文章编号及其原FP32分数。贡献和文章累加公式不变。整条查询计划变成“贡献→累加→分块Top-K→合并”四段，同一Stream中串联，不在两级Top-K之间下载候选或插入CPU排序。

为什么局部K足够：一篇文章如果进不了所属块的前K，那个块内已有K篇文章按相同分数/编号规则优于它，因此它也不可能进全局前K。分块发生在完整文章总分之后，不能对每个词的部分分数先截断。

## 首轮接口与明确限制

新合并算子的IR在 `standalone/topk_merge_operator.json`。这是生成工程的定义，不是手工猜出的aclnn头文件：

- `partial_scores`：FP32 `[Q,S,K]`，来自现有Top-K的输出缓冲。
- `partial_ids`：int32 `[Q,S,K]`，块内编号，每块内部唯一；候选已按分数降序、同分编号升序排列。
- 属性 `k`、`doc_count`：最终K和原文章数。
- 输出 `top_scores` FP32、`top_ids` int32，形状均为 `[Q,K]`。

原文章编号由NPU计算：`块编号 * 每块文章数 + 块内编号`。合并时比较这一原编号，不能把候选槽位当成文章编号。

首轮要求S在1..32、K在1..32、文章数能被S整除、每块文章至少K篇、Q*文章数不超过当前int32组数限制。非整除尾块留作后续扩展，当前拒绝而不丢文章或悄悄补CPU结果。输入来自已经验证的分块Top-K，内核另检查分数有限/非负、编号范围及候选有序性；失败写负值哨兵，让结果检查明确失败。

## 已完成与尚未完成

已完成新算子的op_host、tiling和SIMT kernel、独立IR及工程生成选项；CPU算法模型681组对独立完整排序的编号/分数字节检查通过，并覆盖非法尺寸/候选、接近int32编号上限及生成模板复制/备份/指纹。测试未编译或运行Ascend C，不等于设备正确性或性能收益。

现有贡献、累加、并行Top-K和已安装vendor不改。新vendor固定为 `bm25_topk_merge`，新工程 `results/Bm25TopKMerge`。没有新增Faiss、IndexSDK或其他检索包依赖。

用户已在黄区完成新合并算子的A5编译、打包和独立安装，安装日志为2026-10-15 22:54:54。真实生成头文件已读取：准备接口依次接收两个候选张量、k、docCount、两个输出张量、workspaceSize及executor；执行接口接收workspace、大小、executor和stream。实际新API库的 `ldd -r` 未报告缺失库或未定义符号。这不代表ATC运行库问题已修复。

已据此补上独立合并调用器和七组设备验收工具，代码在本机完成，尚未在黄区编译/运行。尚未完成合并设备正确性、四段调用器和完整性能对照。当前 `benchmark_case.py` 仍测试旧三段版，不会自动切换到新四段实验。

## 现在执行：先验新合并算子

这轮输入是用于单元测试的完整文章分数和各块前K候选。Python在CPU上准备这些已知候选，便于单独核对新合并算子；调用器只读取候选数组，不读取CPU期望输出。验收不代表第一阶段Top-K由NPU执行，也不作BM25性能比较。后续四段流程的候选必须由NPU生成并留在设备。

测试有七种尺寸：S=1/2/3/8/16/32，K=1/3/10/32，含K=全部文章。每组65条查询，覆盖全零、全同分、升序、降序、随机同分、跨块最大值、相邻FP32值及正负零。查询多于前次实测56个工作块，检查循环处理；实际新算子的工作块数应由设备/tiling日志核对，不宣称已测整卡利用率。每组在同一进程运行三次，每次毒化输出、重新取得普通executor、执行、同步、下载，要求输出字节一致。

同步最新Git源码后，校验应为85个文件。当前算子已安装，无需重新生成或覆盖旧vendor。逐条执行，上一条成功后再继续；输出目录已存在就换新名称：

```bash
cd /test/w50066029/bm25-npu
source /usr/local/Ascend/cann-9.2.0/set_env.sh
python3 verify_package.py
set -o pipefail
cmake -S npu_prototype/topk_merge_host -B results/topk_merge_host_build -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 -DBM25_MERGE_OPP_ROOT=/test/w50066029/bm25-npu/results/custom_opp/vendors/bm25_topk_merge 2>&1 | tee results/topk_merge_host_configure1.log
```

配置成功后：

```bash
cmake --build results/topk_merge_host_build -j 2>&1 | tee results/topk_merge_host_build1.log
```

编译成功后，选择空闲卡，避开设备4。下面用设备1：

```bash
python3 standalone/validate_topk_merge.py --output results/topk_merge_validation1 --device 1 2>&1 | tee results/topk_merge_validation1.log
```

工具为子进程单独设置本次合并vendor路径；C++按绝对路径加载其API库，核对两个函数的实际所属库。任何输入指纹、实际头文件、工程源码、设备API、同步、下载或结果核对失败都会停止，不写总成功报告，不用CPU结果填充设备输出。每组实机日志保存为该组的 `device.log`，已通过的组保存 `checked.json`；全部通过后才写 `results/topk_merge_validation1/report.json`。

验收标准：七组实际设备调用全部成功，全文章独立排序得到的编号、顺序及原FP32分数字节与输出一致，且每组同进程三次输出一致。本机已检查七组夹具、损坏输入/输出、尺寸/接口不符及进程失败不出成功报告；使用了测试替身验证编排，未编译C++或运行NPU，不能当作这轮设备验收。

## 当前先生成与编译新算子

以下保留新环境从零生成的步骤；当前黄区已经编译安装，不需重复执行。逐条成功后才继续，输出已存在时换新路径，不删旧工程/旧报告：

```bash
cd /test/w50066029/bm25-npu
source /usr/local/Ascend/cann-9.2.0/set_env.sh
python3 verify_package.py
python3 standalone/create_project.py --operator topk_merge --compute-resource ai_core-ascend950 --compute-unit ascend950 --msopgen-python python3
```

生成成功后：

```bash
cd /test/w50066029/bm25-npu/results/Bm25TopKMerge
set -o pipefail
bash build.sh 2>&1 | tee ../topk_merge_build1.log
```

编译日志需要确认实际ascend950设备代码和安装包生成，遇失败先按实际报错修复，不跳过共享库校验。

## 编译通过后再做的步骤

安装到已有自定义OPP根目录下的新vendor，读取实际生成头文件和库；下列文件名依当前生成器命名规则，若工具实际输出不同先按日志核对：

```bash
bash build_out/custom_opp_openEuler_x86_64.run --install-path=/test/w50066029/bm25-npu/results/custom_opp
cd /test/w50066029/bm25-npu
cat results/custom_opp/vendors/bm25_topk_merge/op_api/include/aclnn_bm25_top_k_merge.h
```

读到真实声明后，接入新的独立四段调用器，按绝对路径加载四个API库并核对八个符号来源；普通executor每轮重新准备，缓冲和workspace只在同步后复用。分块Top-K的额外工作区应按Q*S和每块文章数计算，而非继续按原Q/文章数计算；候选2D/3D描述必须指向同一设备缓冲且尺寸一致。四段完成后才同步/下载最终结果，调试下载单列。

先做新合并算子的边界及正确性，再接完整四段：S=1/2/3/8/16/32、全零、跨块同分、K=1/32、小数据K=全部文章、非整除拒绝、多查询循环、重复输出和CPU直接交叉。输入完整得分后再选候选，不能用CPU产生候选冒充实机第一阶段性能。

性能阶段先比较同一 `normal_64k_q1` 的S=8/16/32与旧NPU及最快已测CPU8线程；增加一个合并算子也会增加工作量，必须测实际总时间，不承诺加速倍数。S=1可以作为保持原单级路径的对照。随后补32查询基线，检查批量下是否需要较小分块数，并继续更多规模、含传输与原版BM25S比较。

## 关键函数

- `ValidShape` / `Tiling`：检查候选形状、整除和编号上限，设置每查询一个完整32线程合并组。
- `MergeTopKSimt`：每线程负责一块有序候选列表，只保存当前列表头；组内比较分数/原文章编号，输出一个赢家，只推进赢家所属列表，重复K次。
- `asc_shfl_xor`：完整32线程组交换当前候选以找最优项；不足32块的线程提供哨兵，仍参加交换，没有失活来源。
- 工程准备工具：使用实际生成模板、备份模板、复制新源码并保存指纹；不会编译、安装或运行设备。
- `DescribeCandidates`：把连续候选数组描述为三维 `[查询数,块数,K]`，不在CPU重新排列候选。
- `ValidateCandidates` / `ValidateSelected`：主机先检查候选合法，再检查设备输出的编号范围、顺序和候选归属；不替设备排序。
- `validate_topk_merge.prepare` / `check`：前者只创建明确标记的CPU单元测试输入，后者用全部文章独立排序核对设备文件，二者不伪造NPU结果。
- `validate_topk_merge.run`：逐组启动实际调用器、保存日志；设备程序内部三次覆写检查与外部独立完整排名检查都成功后才写总报告。
