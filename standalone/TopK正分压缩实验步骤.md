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

这些是模型及主机工具检查，**没有CANN编译，没有设备同步或UB实际行为验证，也没有新Top-K性能结果。** 本轮先完成真实编译及实际接口核对；不把模型、打包或安装当作实机正确性。

## Windows提交：新文件要明确加入Git

```powershell
cd F:\Ascend\bm25\bm25-a5
git add -- npu_prototype/topk_positive_compact standalone/create_topk_positive_compact.py standalone/check_topk_positive_compact_model.py standalone/test_topk_positive_compact_generator.py package_manifest.json README.md 'standalone/*.md'
git commit -m "feat: 增加1024容量正分压缩的独立Top-K实验"
git push
```

## 黄区先生成和编译

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

失败停在第一处错误，暂不安装或改变当前Search运行目录。只分享黄区允许的编译摘要/错误和生成声明，文件不外传。IR与旧版相同不等于可以不看新生成头文件，先核对再准备独立验收与接入。

## 后续验收和保留标准

实际接口确认后，集中准备一次独立设备套件：保留原25组，再重点覆盖1023/1024/1025正分、8192/8193长行边界、少于K个正分、压缩顺序与原ID顺序不同、全零/正负零/次正规、全等与稠密输入；同进程换输入、poison和原始位核对。通过后固定已验证的新评分和相同合并，用原完整Search计时边界作当前Top-K/压缩Top-K六轮交替对照。没有稳定完整Search收益就不切主线，不用候选数量下降替代性能结果。CPU/原版BM25S最终同边界比较仍须完成。

关键函数：`RadixTopKSimt`增加完整校验、正分统计/成员标记和有界压缩分支；`ReadBits/ReadId`区分原数组和片上名单，始终返回原分数字节与真实ID；`WriteSelected`沿用最后32槽排序并从原输入取输出位；`populate`只给独立新工程拷贝三份源码并记录SHA256。

建议commit备注：`feat: 增加1024容量正分压缩的独立Top-K实验`。
