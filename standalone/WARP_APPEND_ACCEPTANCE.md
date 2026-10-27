# Warp聚合直接追加：代码复核与A5验收

日期：2026-10-27。状态：另一模型的新方案已经复核、修复并通过本地检查；尚未实际CANN编译或A5执行。独立候选，不替换Sparse Pack。

## 改了什么

已验收Sparse Pack在评分时，每个文档首次由零变正会原子OR到位图；评分后读取256线程的两段位图，统计数量、做全块前缀，再发射正分文档ID及分数。它已经避免完整8192分的Pack扫描。

新版本在评分的每个warp迭代中，用`asc_ballot`得到首次变正的lane集合，popcount得到各lane的追加位置。只由lane0执行一次UB原子加分配连续段，再用`asc_shfl`广播起点，直接写正分ID列表。后续只按ID列表收集连续分数，省去位图OR、位图读取/计数、Pack全块前缀和逐bit发射。不是在原先两个约2%候选上继续叠加，仍直接对照已验收Sparse Pack。

每个词的倒排文档唯一，词间屏障保留；只有首次从零变正时追加，因此重复查询词仍按原顺序累加，文档不重复追加。游标记录真实数量，只有槽位小于1024才写数组；超过容量时整行走原完整选择，不能截断1024条后返回。全零、少于/等于K、非法数据和原始分数debug2路径仍保留相应精确回退/拒绝行为。

列表因warp调度可乱序。收集器按真实文档ID取每线程同分最小者，再找严格大于上个赢家ID的下一个最小者，保留分数降序、真实ID升序。寄存器阈值、最终32槽排序、全局ID转换和32路合并树不变。乱序同分收集可能多扫描局部列表；ballot/shuffle和游标原子也有成本，收益必须实测。

FP32算式、查询词顺序、8192叶宽、256线程、显式50848字节UB/64KiB动态UB、公开ABI、索引格式、CPU、主机及完整Search计时边界均保持不变。

## 复核时修复的问题

1. 原生成器只把Pack函数的前半段替换成新完整函数，旧后半段残留到函数外，会导致C++编译失败。现提取并替换完整函数；新增完整生成C++括号与函数尾检查，避免“能逆向还原”检查漏掉这个问题。
2. 原评分中的非法文档/非法TF等分支使用`continue`，会跳过当前迭代的ballot/shuffle。现改为嵌套保护分支，非法文档不读docNorm，错误lane和尾部不活跃lane均到达本warp集合点，包括lane0。正常输入的FP32运算链逐字保留。
3. 54项原始分数检查会走debug2，不能单独验证新正常评分追加路径。现增加正常评分K1/10/32各一组8193篇对抗用例，基线/候选共六个进程，覆盖32/1023/1024/1025正分、同分、重复词、全零、末尾单文章叶和容量溢出回退。跨规模复用必须同时具有三种K的匹配完整输出证据。

本地已通过376组追加范围模型、60组重叠/重复词FP32和随机warp调度、144组乱序严格收集、72组K/容量边界，以及生成结构、源码逐字逆向还原、独立vendor、头文件发现与七档调度检查。模型不模拟950硬件调度或编译器，不替代设备证据。

复核记录、当前三个新源码文件哈希及官方[ballot](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/Warp_functions/Warp_vote_functions/asc_ballot.md)/[shuffle](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/Warp_functions/Warp_shfl_functions/asc_shfl.md)依据见[复核证据](warp_append_review_20261027.json)。官方接口资料和本地结构检查不能证明黄区CANN9.2.0编译成功，实际编译仍是第一道门。

## 黄区执行

先提交并同步完整工作区修复；原先暂存的三个新文件早于本次修复，提交时必须包含当前工作区内容。无需重跑旧Sparse Pack、warp流水或自适应清零。保留`results/sparse_pack_scales1`及其引用的所有旧输入、安装、二进制和证据。

```bash
cd /test/w50066029/bm25-npu
set -o pipefail
python3 verify_package.py && \
python3 -u standalone/warp_append.py --output results/warp_append_scales1 2>&1 | tee results/warp_append_scales1.console.log
```

结果目录`results/warp_append_scales1`和独立vendor `bm25_fused_score_topk_warp_append_large`应未存在；脚本拒绝覆盖旧结果或安装。沿用此前已成功的CANN9.2.0环境与设备5，自动检查磁盘、主机和设备内存预算，不更改CPU代码、线程、亲和性或NUMA设置。

默认先测262144篇，再测1024/8192/65536/1048576/4194304/10000000，保留七档共204个正式性能进程、12240次正式Search。一处编译安装与受影响检查完成后，其余档按源码/安装/输出哈希复用。小库落后继续覆盖。26万篇不会因为落后CPU而跳过其他规模。

每档固定Q8、3词、K10、六轮正逆顺序，每进程5个A/B预热周期、30个正式周期。NPU、Sparse Pack与冻结CPU对手用同一输入，每轮取声明CPU配置较快者；65K及以下继续包括相同CPU二进制的1/1和32/32配置。计时仍从原始词ID进入Search到CPU收到严格Top-K。

本轮保留工作量/内存摘要输出：索引字节、倒排数、TF/DF、各查询命中记录/文章数、叶正分分布、设备空闲探测和计划内存界限。它们不是实测HBM流量、实际warp操作计数或整卡利用率。

## 怎样验收

- 先通过实际编译安装、54项原始选择、三种K正常评分对抗检查，再接受任何性能判定。分数/树输出位、严格排名、索引不变、保护区和重复结果均由既有检查链验证，不能以CPU代答。
- 增量对照必须是本轮同期Sparse Pack。保留原预声明规则：A/B各6/6更快、中位耗时降低至少5%、六轮P95中位数更低，并通过同期声明CPU配置门槛；5%是候选替换规则，不是MT业务要求。
- 本轮26万篇CPU胜负单独报告。历史较快CPU约66µs，而NPU约82µs；如果本轮CPU回到约100µs慢态，即使获胜，也不能称已稳定超过历史快态。历史时间只作上下文，不替代同期CPU样本；不通过改变CPU配置获取NPU优势。
- 最终`NPU warp聚合直接紧凑全规模覆盖`中，`stable_cpu_winning_docs`和`retained_candidate_docs`分开解释。退出0只代表七档覆盖完成，不代表全部更快。不将跨会话CPU波动归因于NPU优化。

完成后提供`NPU warp聚合工作量`、`NPU warp聚合内存预算`、各档完整Search和六轮判定、最终全规模汇总。若失败，保留目录，提供首次错误及对应日志；不要删除重跑或只换目录去覆盖既有vendor。当前用户没有MT认可业务数据，结论仍限固定均匀合成分布和查询条件。
