# 完整Search：只替换评分实现的新旧对照

## 先理解本轮做什么

此前诊断：旧评分约266us、局部Top-K约271us，合并较小。定位已经完成，本轮不重新运行msprof，也不再做独立评分边界补验。新评分已在设备2通过小样本、4097文章及三组64词/多分区补验；这些证明指定评分输入正确，尚未证明完整Search正确或更快。

本轮输入是已有`results/search_256k_q8_input1`：262144篇文章、8条原始查询词编号、K=10，A为原查询，B为同长度轮换词编号。索引仍是同一组原始TF/IDF/长度项倒排数组，按查询算公式，没有换成预计算贡献。这个数据是合成正常语料，不是真实业务数据。

处理步骤在一个脚本中连续完成：

1. 核对已经通过的三组新评分报告、实际执行程序/日志/完整输出/索引/源码；核对旧评分、新评分、已有Top-K/合并与安装产物。
2. 在新输出目录下建立两个独立运行目录。两边的局部Top-K和合并都指向相同安装库，评分入口分别指向旧`bm25_query_score`和新`bm25_query_score_range`。仅为目录别名，不复制库、不修改旧vendor。每个版本独立进程，只给子进程设置它需要的三个库路径。
3. 使用已有`bm25_search_runner`，对旧、新两版各自动执行A/B/A/空/B五次完整Search。全部文章分数、局部候选、最终编号/顺序/原始FP32分数字节、重复输出及索引回读必须通过，才能开始计时。出错立即停止，不换CPU结果，不回退其他评分版本。
4. 使用同一个已有`bm25_npu_search_benchmark`，同一张空闲卡，六轮独立进程交替旧/新、下一轮新/旧。每进程5个预热周期、30个正式周期，A/B在周期内也轮换顺序。保留每条样本、每进程中位数/p95和逐轮旧版/新版比值；原查询A和变化查询B分别报告。

计时仍为：CPU已有原始分组词编号 → 开始主机计时 → 验证/整理查询、上传、设备查常驻索引、评分/Top-K/合并、同步、下载和必要结果检查/请求资源释放 → CPU持有合规Top-K → 停止。索引加载、冷初始化/容量空查询预启动单列；完整分数/局部候选调试下载、输出污染和独立公式核对在计时外。沿用同一套C++入口与计时边界，不是纯评分内核或resident结果。

最终输出`report.json`包含两版完整Search门禁、六轮A/B时间、每轮四个前后检查点的直接交叉及全部来源指纹。运行前后完整输出逐字节一致；每条计时调用的Top-K仍由已有C++程序核对。只在所有检查完成后写成功状态。没有新C++或新内核，**无需再次编译或安装**。

## 支持范围和比较边界

这次独立入口为复用现有计时二进制，明确限制**整个查询批次最多64词**。现有C++ `PackQueries`会在每次Search计时内检查整批总词容量；因此每条查询也必然不超过新评分的64词上限，没有把必要保护移出计时。两版本使用完全相同的容量：从原输入得出，256K/Q8是8查询/24词；不是给新版本单独缩小容量。更大总词数的批次需另行加入计时内逐行64词保护，脚本先拒绝，不能绕开这一限制。本轮不宣称已实现任意批量的新评分通用Search接口。

8条查询整批时间不是单条查询延迟，报告中吞吐量用同批查询数和平均Search时间计算。旧评分/新评分都在NPU上，这轮比值是实现变化带来的收益，不能命名为NPU对CPU硬件加速。CPU及原版BM25S仍是后续完整Search对手；保留已有CPU正式结果，不用历史CPU时间和本轮新NPU时间直接拼出正式加速比。

运行目录使用Linux符号链接，C++按绝对路径加载库并检查符号实际归属；自定义算子搜索路径同样解析到真实目标，避免同时注册两个评分实现。本机Windows缺少创建符号链接权限：本轮主机测试对链接对象和子进程作模拟，不声称实际Linux别名及CANN运行已通过。

## Windows提交：新增文件必须加入Git

此前清单先提交、三个脚本未跟踪导致黄区缺文件；这次明确加入两个新脚本和说明，再提交清单及已更新说明：

```powershell
cd F:\Ascend\bm25\bm25-a5
git add -- standalone/compare_search_score_versions.py standalone/test_search_score_versions.py package_manifest.json README.md 'standalone/*.md'
git commit -m "feat: 接入连续分区评分并比较新旧完整Search耗时"
git push
```

## 黄区：一条入口自动核对后再对照

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -o pipefail
```

校验通过后：

```bash
python3 standalone/compare_search_score_versions.py --input results/search_256k_q8_input1 --output results/search_range_256k_q8_pair1 --device 2 2>&1 | tee results/search_range_256k_q8_pair1.log
```

默认复用已验收的`results/search_host_build/bm25_search_runner`、`results/search_benchmark_host_build/bm25_npu_search_benchmark`、`results/query_score_range_boundaries_check1`、`results/radix_collect_topk_validation1`。对应路径改过名字时使用`--diagnostic-runner/--runner/--boundary-acceptance/--topk-acceptance`显式指定，不能跳过验证。输出目录已存在就换新编号，保留旧证据。文件继续留黄区，只分享允许的终端汇总。

成功后可在黄区打印短摘要：

```bash
python3 -c "import json; r=json.load(open('results/search_range_256k_q8_pair1/report.json',encoding='utf-8')); print(r['status']); [print(k,v) for k,v in r['paired_summary'].items()]"
```

## 关键代码和保留标准

`validate_boundaries`复核已有评分设备证据；`create_runtime/validate_runtime`隔离并核对实际加载目标；`preflight`执行两版完整Search公式及严格排名门禁；`bench.run_one`复用原二进制、计时和检查器；`pair_summary`分别汇总A/B各轮中位数及新版本更快轮数。原C++ `NpuSearchSession::Search`承担完整实际查询，源码和编译指纹不变。

先判断完整Search是否有一致改善。候选更少或二分次数下降都不是速度证据；某一轮最低值也不是保留理由。如果新版本没有稳定改善或变慢，保留旧版主线、停止该评分实验，转向已有证据中的局部Top-K瓶颈。若完整Search改善成立，再考虑与合理CPU的同边界复测；仍有无法解释的耗时才针对性采集，不重复原来的整体瓶颈分析。

本机49项证据来源/完整Search门禁/新旧顺序/计时/严格结果/失败即停检查通过，原完整Search计时工具55项回归也通过。测试使用真实主机输入准备和独立公式，但安装/二进制/链接对象/设备子进程均为明确模拟；没有新C++编译，也不证明Linux链接或NPU性能。实际完整Search门禁、别名加载与六轮性能等待黄区执行。建议commit备注：`feat: 接入连续分区评分并比较新旧完整Search耗时`。
