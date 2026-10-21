# 局部Top-K：有界正分名单压缩实验

## 先用简单的话说明

新评分已经让256K文章、8查询整批完整Search从约0.583ms降到0.394ms，六轮全部更快；评分固定保留。本轮只尝试改局部Top-K，不重新做266us/271us那张瓶颈采集。

用户已核对刚才保存的新评分完整输出及SHA256，只在黄区打印统计，没有导出文件：

| 查询组 | 每块文章数 / 块数 | 正分比例 | 每块正分最少 / 中位数 / 最多 | 1024容量覆盖 |
|---|---|---|---|---|
| A | 8192 / 256 | 8.91% | 643 / 730 / 800 | 256/256 |
| B | 8192 / 256 | 8.92% | 673 / 730 / 825 | 256/256 |

所以第一版用1024容量，不用装不下当前任一分块的512，也暂不多占用2048容量空间。这只是当前样本的容量依据，不能假定以后查询都满足，也不等于速度证据。

输入仍是同一个FP32文章分数数组和K，输出仍是分数降序、同分原文章编号升序的K项及原始FP32分数。公式、索引、评分、分块数和最终合并都不改。

处理分三种情况：

1. 行长1025～8192时，先扫描一遍校验全部分数，统计正分数。每个线程用两个32位标记记住自己读到的哪些位置为正分，最多覆盖64个位置；只把这些位置的原分数字节和原文章编号读入片上小名单。
2. 正分数不超过1024且多于K：继续使用当前11/11/10位三轮精确阈值选择及候选收集，不过三轮只读片上正分名单。通常约730个参与，替代每轮8192个；最后仍只排序K项及32槽补齐。
3. 正分数不多于K：这些正分必须全部保留，再从零分文章按编号升序补足，最后按规则排序。正分超过1024，或行长不在上述范围，则在同一个NPU算子内走保留的完整精确选择路径。**不截断，不返回近似结果，不调用CPU或另一个算法包。**

“算子内完整路径”是预定的合法算法分支，不是设备失败后的回退。非法分数仍返回错误标记，未来调用器必须停止。新算子暂未接主线，旧vendor和全部实际验收证据保留。

## 为什么借鉴这条思路

| 来源 | 已核对的事实 | 本实验使用与区别 |
|---|---|---|
| 组内IndexSDK，提交59213c4ca5ebb643c031c8b819e90e3a5cc21a8c，`feature_retrieval/src/ascendfaiss/ops/ascendc/op_kernel/cagra_rabitq_distance.h::update_histogram`，约687～813行 | 输入`x`为UB数组，使用状态位跳过后续已排除记录；三轮11/11/10统计 | 保留当前自研三轮阈值选择，借鉴减少后续参与者；不复制其固定内存假设或原子等值名额规则 |
| [DeepSelect固定源码common_parts.cuh，8e70df71](https://github.com/deepseek-ai/DeepSelect/blob/8e70df71d2a4b0c969ef96dc3b8998efa09a3315/csrc/cuda_kernels/common_parts.cuh)，及[设计说明](https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.md) | 采用候选值/编号缓冲和基于直方图的重建；设计说明描述先过滤再压缩，并用成员位标记减少再次遍历 | 只借鉴候选缓冲与标记思想。这里用BM25非负分数的零边界做一次精确过滤，**不是完整复现DeepSelect**；不引入随机遍历、PTX、线程块集群或GPU性能结论 |
| [DeepSelect支持范围](https://github.com/deepseek-ai/DeepSelect#supported-cases) | 公开说明Ascend实现只支持BF16，FP32为CUDA路径 | 我们自研FP32整数键和原始位搬运，不降精度，也不运行该库 |
| 已验证的`topk_radix_collect` | 三轮精确阈值、一次候选前缀、同分有序头归并、只整理最终K项 | 主体沿用，不重新发明选择算法；只让它能读有界片上正分名单，并按真实文章编号比较 |
| 原零分计数汇总实验 | 同期六轮全慢，约慢2.5% | 不重新做“同样扫描整行，只换零分桶计数”的微调。这次改变参与选择的输入集合，仍需实测成本 |

外部源码仅为参考，不进入工程运行依赖。上表固定CUDA文件和当前公开说明分开记录；没有将文档固定链接的旧提交当作仓库当前HEAD。

## 正确性中最容易错的地方

正的次正规数也必须算正分，用原始位判零，不能加epsilon。正零与负零只在比较键里视为同分，输出从原GM数组复制，保留原始字节。

压缩后的位置不能代替文章编号。例如压缩顺序可能是文章128、文章1；同分时必须选文章1。各线程保留自己原来编号递增的正分序列；门槛同分的头部归并比较真实ID，并通过名单位置推进该线程的游标。最终局部编号仍是原分块内编号，已有跨分块合并规则不变。

容量先统计、再判断、最后写入；1025个正分不会被截成1024。行长8193及以上不用64位成员标记，直接走完整路径。每个查询重新初始化计数、成员标记和选中槽，不能读上条查询留下的小名单内容。NaN、Inf或负分始终不被忽略。

## 代价与未知项

新增整行预筛、成员标记、一次线程计数前缀及正分搬运。显式算法UB从8512变为16704字节（额外8192用于1024组原位/ID），SIMT局部预留仍32KiB；host按真实平台检查合计49472字节是否可用，不能假定950容量。正分容量溢出的1025～8192长行会多做预筛；短行和大于8192长行不做预筛，但代码/UB配置仍可能影响性能。

目前仅能说三轮统计的参与项从8192变成约730，**不能宣称11倍加速**。片上整数原子争用、增加的位运算、UB占用、分支与同步都可能抵消收益。现有271us是旧Top-K的采集诊断，也不能全部归因于某一个循环。

## 本机完成了什么

新增独立`npu_prototype/topk_positive_compact`三份源码和`standalone/create_topk_positive_compact.py`。使用原`topk_operator.json`的输入输出定义、独立工程`Bm25TopKPositiveCompact`和独立vendor `bm25_topk_positive_compact`。旧生成器、算子、C++调用器、评分、验收脚本、BM25S原版及二进制样本都不改。

`check_topk_positive_compact_model.py`的899项整数/线程模型与独立完整数值排序通过：1023/1024/1025正分容量、1024/1025和8192/8193行长分界、128线程尾部、成员标记低/高位含31/32/63、同分原编号反例、K=1/10/32、正负零原字节、次正规数、最大有限数、同一缓存稠密/稀疏/全零轮换和非法分数。47项假模板/备份/元数据/拒绝保护/任务覆盖检查也通过。

上述本地检查只是模型及主机工具检查。**随后用户在黄区已完成950实际编译和打包**：独立vendor正确，设备目标`Bm25TopK_23cbb04723633b70cf874275924b41b4.o`生成，安装包生成；两份实际自动生成头文件已读到并核对。尚未安装此vendor或完成设备正确性验收，没有新Top-K性能结果。

实际准备接口为`aclnnBm25TopKGetWorkspaceSize(scores, int64_t k, topScoresOut, topIdsOut, workspaceSize, executor)`，执行接口为`aclnnBm25TopK(workspace, workspaceSize, executor, stream)`；四个张量参数为`const aclTensor*`，其余实际类型由下述调用器直接包含安装头文件并静态核对。没有用IR相同代替实际声明核对。

现在新增独立`topk_positive_compact_host`和`validate_topk_positive_compact.py`。主机沿用已验证的ACL资源、绝对路径加载及逐次全排序核对逻辑；不改旧调用器。旧调用器会在记录中写8512字节算法UB，不能直接用作新算子的正确验收记录。新调用器改为16704，并把实际编译时三份主机源码SHA256写入报告，拒绝拿旧可执行文件配新脚本。生成器元数据的`project_generated_not_compiled_not_device_verified`仅记录生成时状态，不能推翻后续真实编译日志，也不是运行证据。

### 本次集中验收怎么做

输入是人工构造的FP32分数矩阵；设备输出每行K个编号及分数。保留旧25组输入，再加12组：行长1024/1025/8192/8193分别配K=1/10/32，每组65行包含0、K-1/K/K+1、730、1023/1024/1025个正分、全稠密、同分原ID反例、次正规数、最大有限数、正负零。超过容量必须保留正确完整路径，不能截断。

每组同进程A/B/A/B/A五次，每次先把输出覆写成NaN和-1，再实际上传、准备、执行、同步、下载；C++完整数值排序逐个核对编号及原始分数字节，并检查A/B重复字节一致。Python重新完整排序复核保存的两组输出。安装头/API库/950设备目标均须匹配工程产物，源码及输入在运行前后必须不变。全部37组通过，才允许进入完整Search对照；任何失败保留报告并停止，无CPU替代输出。

`boundary_rows/prepare_boundary`组织容量及同分输入；`validate_install`检查实际声明及安装产物；`validate_record`核对实际调用次数、覆写检查和编译源码；`run`依次运行37组并保存每组输入/日志/设备输出指纹。原C++中的`FullSortIds`只生成核对答案，`CheckFullOutput`只验证下载结果，不写入设备输出文件。

本地57项工具/失败保护检查通过，包含全部37组实际输入准备和Python完整排序；其中调用器和设备产物均为明确的假文件、子进程被模拟，**不是C++编译或设备验收**。原899项算子模型和47项生成器检查继续保留。

### 提交本次验收工具

```powershell
cd F:\Ascend\bm25\bm25-a5
git add -- npu_prototype/topk_positive_compact_host standalone/validate_topk_positive_compact.py standalone/test_topk_positive_compact_validation.py package_manifest.json README.md 'standalone/*.md'
git commit -m "test: 增加正分压缩Top-K的37组独立设备验收"
git push
```

### 黄区安装、编译调用器、一次运行全套

已完成的算子工程不重新生成、不重新编译；仅安装其现有产物。以下逐条成功再执行下一条。运行目录已存在就改用新名字，保留旧日志。

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -o pipefail
bash results/Bm25TopKPositiveCompact/build_out/custom_opp_openEuler_x86_64.run --install-path=/test/w50066029/bm25-npu/results/custom_opp
```

```bash
cmake -S npu_prototype/topk_positive_compact_host -B results/topk_positive_compact_host_build -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 -DBM25_OPP_ROOT=/test/w50066029/bm25-npu/results/custom_opp/vendors/bm25_topk_positive_compact 2>&1 | tee results/topk_positive_compact_host_configure1.log
```

```bash
cmake --build results/topk_positive_compact_host_build -j 2>&1 | tee results/topk_positive_compact_host_build1.log
```

```bash
python3 standalone/validate_topk_positive_compact.py --output results/topk_positive_compact_validation1 --device 2 2>&1 | tee results/topk_positive_compact_validation1.log
```

脚本给子进程只设置新vendor，绝对路径加载API并核对符号归属；不用在当前终端混合source多个同名Top-K。无需再次采集旧评分或扫描CPU线程。`results/topk_positive_compact_validation1/report.json`记录37组状态；当前只有工具准备通过，以上新的主机编译及设备验收仍待黄区执行。黄区原文件留黄区，仅分享允许的错误或末尾摘要。

## Windows提交：新文件要明确加入Git

```powershell
cd F:\Ascend\bm25\bm25-a5
git add -- npu_prototype/topk_positive_compact standalone/create_topk_positive_compact.py standalone/check_topk_positive_compact_model.py standalone/test_topk_positive_compact_generator.py package_manifest.json README.md 'standalone/*.md'
git commit -m "feat: 增加1024容量正分压缩的独立Top-K实验"
git push
```

## 黄区生成和编译（已完成，以下仅保留复现步骤）

逐条成功再继续。新目录已存在就停止并保留日志，不能覆盖原工程：

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -o pipefail
python3 standalone/create_topk_positive_compact.py --compute-resource ai_core-ascend950 --msopgen-python python3
```

生成成功后：

```bash
cd /test/w50066029/bm25-npu/results/Bm25TopKPositiveCompact
bash build.sh 2>&1 | tee ../topk_positive_compact_build1.log
```

编译成功后读取实际生成的接口：

```bash
find build_out -type f -name aclnn_bm25_top_k.h -print -exec cat {} \;
```

本阶段已完成；当前执行上方“黄区安装、编译调用器、一次运行全套”，不重复生成工程。旧Search运行目录不变。只分享黄区允许的编译摘要/错误和生成声明，文件不外传。IR与旧版相同不等于可以不看新生成头文件，先核对再准备独立验收与接入。

## 后续验收和保留标准

实际接口确认后，集中准备一次独立设备套件：保留原25组，再重点覆盖1023/1024/1025正分、8192/8193长行边界、少于K个正分、压缩顺序与原ID顺序不同、全零/正负零/次正规、全等与稠密输入；同进程换输入、poison和原始位核对。通过后固定已验证的新评分和相同合并，用原完整Search计时边界作当前Top-K/压缩Top-K六轮交替对照。没有稳定完整Search收益就不切主线，不用候选数量下降替代性能结果。CPU/原版BM25S最终同边界比较仍须完成。

关键函数：`RadixTopKSimt`增加完整校验、正分统计/成员标记和有界压缩分支；`ReadBits/ReadId`区分原数组和片上名单，始终返回原分数字节与真实ID；`WriteSelected`沿用最后32槽排序并从原输入取输出位；`populate`只给独立新工程拷贝三份源码并记录SHA256。

建议commit备注：`feat: 增加1024容量正分压缩的独立Top-K实验`。
