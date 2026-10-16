# Radix四段正确性通过后的固定条件复测

用户黄区日志已确认：Radix四段在6篇文章小样本及正常65536篇文章/单查询/K=10/S=32上实际运行正确，后者与已有同输入CPU结果交叉通过。10个文章编号及次序严格相同；贡献最大绝对差4.76837158203125e-7，完整总分和Top-K分数最大绝对差9.5367431640625e-7，均在原容差内。首末六数组和局部/全局完整排序核对也通过。

本次3轮NPU批次中位数0.04001ms，即40.01微秒。这是正确性阶段附带计时，不与旧CPU或旧Warp时间拼接计算加速比。

## 这一步做什么

输入固定为原`results/normal_64k_q1`，CPU和NPU都使用相同TF、长度项、IDF、分组及结果规则。NPU固定32分块，CPU固定现有8线程C++参考，不同时改算法或挑选分块数。

做6轮独立进程复测。第1、3、5轮CPU先跑，第2、4、6轮NPU先跑；每一方每轮都新启动进程，先做规定的预热，再正式计时。每次指定预热5次、正式测量30次，原NPU调用器还保留额外一次普通预热。所有初始化、文件读写和结果核对均在各自正式计时之外。

CPU测贡献→累加→Top-K，NPU测四段接口准备、提交及同步。两边原始输入已准备好并在各自内存中；这里不含NPU上传下载，NPU提交与同步也不等于纯内核时间。

输出每轮CPU/NPU日志、原始计时、p95、实际输出、同轮CPU/NPU结果交叉报告及`checked.json`，最后保存`report.json`和便于查看的`summary.json`。每轮都要正确，否则停止，保留部分日志，不生成总通过报告。NPU先跑时，也必须等到本轮CPU完成后再配对，不拿上一轮CPU时间代替。

## 本次代码内容

仅修改`standalone/confirm_sharded.py`复测编排，没有修改C++调用器、算子或CPU程序，所以无需重新编译。

- `run(..., order_policy='cpu_first')`保留旧默认行为；新参数`alternate`按轮号轮换谁先启动。每轮保存实际`process_order`，总报告保存`cpu_npu_order`和顺序说明。
- `execute_cpu`启动本轮CPU，核对其参数、原始计时与结果，并检查各轮输出字节一致。
- NPU逐个配置执行后立即检查自身记录；暂存计时，等同轮CPU也完成后，调用`cross_check`比较全部结果并计算同轮CPU时间/NPU时间。
- 原二进制、输入和安装库的复测前后检查、失败退出、原始样本与p95保存规则继续保留。

本地原有默认顺序回归通过；新增6轮/18个模拟子进程检查通过，确认CPU/NPU各先跑3轮、同轮比值正确，并验证子进程失败、计时或结果异常时不会生成总通过报告。模拟进程不是CPU或NPU性能数据；真实性能待下面黄区命令运行。

## 黄区执行

提交、同步本次Python及说明后，逐行执行。现有CPU/NPU二进制和Radix运行目录继续使用：

```bash
cd /test/w50066029/bm25-npu
source /usr/local/Ascend/cann-9.2.0/set_env.sh
python3 verify_package.py
set -o pipefail
python3 standalone/confirm_sharded.py --input results/normal_64k_q1 --output results/radix_64k_q1_s32_balanced1 --cpu-runner results/cpu_host_build/bm25_cpu_runner --npu-runner results/radix_sharded_host_build/bm25_sharded_benchmark --vendor-dir results/radix_sharded_runtime/vendors --threads 8 --shards 32 --device 2 --rounds 6 --warmup 5 --repeats 30 --order-policy alternate 2>&1 | tee results/radix_64k_q1_s32_balanced1.log
```

保持设备空闲、不同时开启msprof；CPU配额和其他任务干扰仍需记录。可改成当前空闲的同规格卡，避开设备4。输出目录必须新建。

完成后直接在黄区查看精简报告，只提供允许分享的终端摘要，文件留黄区：

```bash
python3 -m json.tool results/radix_64k_q1_s32_balanced1/summary.json
```

## 怎样判断结果

先确认6轮全部正确，再看每轮CPU/NPU中位数和p95。报告中的比值为CPU时间除以NPU时间，大于1表示该轮NPU更快，小于1表示该轮CPU更快。看全部轮次和波动，不只挑最好一轮，也不把平均每条耗时误写成实测单条延迟；此输入本来就是一条查询。

若各轮稳定更快，只能先称“此输入、此8线程CPU参考、常驻准备数据计算条件下更快”。不能直接推广成整机最优CPU、完整查询或原版BM25S均被超过。

当前CPU源码中，贡献、累加可多线程，而Top-K按查询并行：此处只有一条查询，Top-K主要由一个线程处理。8线程不表示Top-K也被8线程分担，更不表示整机只有8核。后续还需研究合理CPU并行/线程配置，并保留原版BM25S独立对手。

有稳定复测结果后，再用同样输入和边界对旧版、Warp、Radix做同期比较；根据实际差距决定是否补设备分段诊断。含传输、批量吞吐、扩大数据规模、常驻索引到完整查询以及BM25S对照仍属于后续验收，不由本轮结论替代。

建议commit备注：`feat: 支持BM25 CPU与NPU复测顺序轮换`
