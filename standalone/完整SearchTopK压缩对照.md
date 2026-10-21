# 完整Search：固定新版评分，只替换局部Top-K

**当前状态：用户已在950设备4通过正分压缩Top-K的37组独立验收，共185次实际调用。** 原25组回归和12组容量/行长边界全部通过，每次输出污染、完整排序、原ID及FP32原始位核对通过。安装及接口加载也已完成。本轮准备完整Search对照工具，不要求重跑37组，也不重做评分采集；完整Search正确性与性能仍待实际运行。

## 用简单的话说明本轮

输入还是已经准备好的262144篇文章索引、8条查询词编号和K=10。索引提前留在NPU；每次Search只上传查询。评分固定为已经验证的新连续分区评分，最后的合并也固定，只比较“原局部Top-K”和“先收集正分再选Top-K”的版本。

输出是两版完整结果核对、每次Search耗时、各轮中位数和p95，以及原查询A和变化查询B分别统计的六轮对照。单批8条查询的时间不能当单条延迟；吞吐按完整Search样本计算。当前仍为合成正常语料，不是业务真实语料。

已有数据每块8192篇、正分约730个，新版可能减少后续参与选择的数据，也增加预扫描、压缩及同步。独立正确并不说明完整Search更快，不能从候选数量推算加速。

## 步骤和计时边界

1. 只读复核已有三组评分边界报告和本次37组Top-K报告的安装产物、源码/调用器指纹、输入、设备记录、输出和日志；重新用CPU排序核对已有Top-K文件，不执行设备用例，不将其时间加入性能样本。
2. 在新输出目录建立两个隔离的运行目录。二者评分入口都指向`bm25_query_score_range`，合并都指向`bm25_topk_merge`；局部Top-K分别指向`bm25_topk_radix_collect`和`bm25_topk_positive_compact`。使用Linux目录别名，不改变旧vendor。现有C++入口仍按原规范路径加载实际库并核对符号归属；每个进程只设置其需要的三个vendor。
3. 使用同一个已有`bm25_search_runner`，每版先执行A/B/A/全空/B五次完整Search，检查评分、局部候选、最终排名、重复输入、索引回读和全部原始分数字节。评分完全相同，因此新旧五个输出数组还须逐字节一致，不能只比较最终十个编号。
4. 全部正确后，再使用同一个已有`bm25_npu_search_benchmark`。六轮独立进程交替“原/新、下一轮新/原”；每个进程5个预热周期和30个正式周期，A/B也交替。计时前后四个检查点逐数组检查，全部样本保留，失败立即停止。

两版都测：CPU已有原始分组词编号 → 开始计时 → 入口验证/查询整理/上传 → NPU查常驻索引、评分、局部Top-K及合并 → 同步/下载/必要结果检查和请求资源释放 → CPU持有合规Top-K → 停止计时。索引加载、初始化、容量分配及冷启动单列；完整分数和候选的调试下载、输出污染、独立公式核对不在正式Search计时内。未将查询相关准备移出计时。

新Top-K没有用户GM工作区，仍使用平台返回的系统工作区；旧入口会在每次准备阶段核对实际大小，不能静默扩容或猜测。若实际大小与既有验收不符，停止并记录错误。本对照复用此前入口的安全范围：整批最多64个词，故每条也不超过评分上限64词；实际Q8是24词，查询长度等原入口检查仍在计时内，不代表已支持任意大批次。

## 执行

本次只新增两个Python文件，旧算子、主机源码与已验收脚本不改；不需要重新生成、编译或安装。

Windows明确加入新文件再提交：

```powershell
cd F:\Ascend\bm25\bm25-a5
git add -- standalone/compare_search_topk_versions.py standalone/test_search_topk_versions.py package_manifest.json README.md 'standalone/*.md'
git commit -m "perf: 增加固定新版评分的Top-K完整Search配对对照"
git push
```

黄区拉取后，使用已有输入和实际程序：

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -o pipefail
```

```bash
python3 standalone/compare_search_topk_versions.py --input results/search_256k_q8_input1 --output results/search_topk_compact_256k_q8_pair1 --device 4 --shards 32 --rounds 6 --warmup 5 --repeats 30 2>&1 | tee results/search_topk_compact_256k_q8_pair1.log
```

`--device 4`沿用本次验收卡号；如果有其他工作占用，改为空闲的同规格卡，两版都用这同一张卡。已有结果目录不覆盖，换新名字。既有37组目录默认`results/topk_positive_compact_validation1`，评分边界目录默认`results/query_score_range_boundaries_check1`，原Top-K目录默认`results/radix_collect_topk_validation1`；实际名称不同用`--compact-acceptance/--boundary-acceptance/--topk-acceptance`明确指定，不重新造一份通过报告。

报告位于`results/search_topk_compact_256k_q8_pair1/report.json`。成功状态须为`complete_search_topk_versions_paired_and_all_checks_passed`；这表示实跑及核对完成，不表示新版一定更快。黄区文件保留黄区，只分享允许的末尾汇总或第一处错误。

## 怎样决定保留

先要求公式、所有局部/最终编号与顺序、原分数字节、跨调用复用全部正确。再看A/B六轮中位数、p95、逐轮原/新比值及波动，不能挑最快一轮或混用历史约0.394ms作为同期对手。

若完整Search没有稳定收益或主要输入退化，保留实验记录，继续使用原Top-K；若收益稳定，再固定获胜组合，回到同边界合理优化CPU及原版BM25S对照，并扩展单条查询/批量/不同数据分布。此次六轮只说明同一NPU更换Top-K的影响，不等于超过CPU/BM25S，不用旧271us采集数计算本次加速比。

## 对应到代码

`validate_compact_acceptance`核对已有37组证明及全部结果，返回真实验收工作区和证据指纹；不调用NPU。`run`固定相同评分/合并、建立运行别名，先调用已有五批Search检查，再交替启动同一个计时程序。`previous.preflight/bench.run_one`沿用此前实际通过的完整Search流程与计时检查。旧脚本只被导入使用，字节不修改。

本地60项来源、输出篡改、配对顺序、计时、传输边界及失败停止检查通过；测试使用真实小输入和独立参考，但安装文件、可执行文件、目录别名和子进程为明确的模拟，不能作为Linux符号链接、C++编译或设备性能证据。本次真实37组设备正确性结论来自用户黄区日志；本脚本的完整Search对照尚待黄区执行。
