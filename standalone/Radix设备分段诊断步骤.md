# Radix四段设备耗时诊断

## 已有性能证据及范围

设备2、65536文章、单条查询、K=10、S=32，CPU8线程参考与Radix做了6轮顺序轮换复测，每个进程指定预热5次、测量30次，6轮正确性全部通过。NPU中位数5轮更快，第6轮CPU更快；用户追加的p95如下，单位微秒：

| 轮次 | CPU中位数 | NPU中位数 | CPU p95 | NPU p95 |
|---|---:|---:|---:|---:|
| 1 | 55.043 | 40.321 | 76.625 | 41.933 |
| 2 | 57.166 | 45.689 | 82.584 | 48.984 |
| 3 | 76.164 | 40.961 | 83.074 | 42.995 |
| 4 | 44.847 | 40.891 | 70.165 | 52.789 |
| 5 | 52.028 | 42.994 | 76.614 | 45.959 |
| 6 | 42.714 | 47.472 | 67.982 | 49.835 |

每轮p95均是该轮30次测量的nearest-rank结果，不是把各轮p95再汇总出的“所有样本p95”。NPU在6轮的p95均较低，支持保留Radix候选；仍不能表述成轮轮中位数取胜或完整查询、最佳CPU、BM25S均已超过。

当前只比较准备好并常驻各自内存的数据计算，NPU侧含接口准备、提交及同步，不含上传下载。当前CPU的单查询Top-K按查询分配，不能视作8线程均分这一段。服务器负载、CPU配额等对波动的影响尚未定位。

## 下一步的输入、处理、输出

先不改内核，用原输入和原四段执行方式加一层msprof采集，直接看设备实际在每一段花了多少时间：

| 自研算子 | 这次处理的输入 | 输出 |
|---|---|---|
| Bm25Contribution | 6034条TF/长度项/IDF | 6034条贡献 |
| Bm25Accumulate | 贡献及65537个分组边界 | 65536篇文章总分 |
| Bm25TopK（Radix） | 32份，每份2048篇文章分数 | 每份10个候选及局部编号 |
| Bm25TopKMerge | 32份候选 | 全局10个分数和文章编号 |

四段仍共用同一stream、整链一次同步。采集首次1轮、检查预热5轮、普通预热1轮、正式10轮；如果数据完整且每段一个设备任务，分析器应交叉核对68条BM25任务，只统计最后10轮。

输出每段正式耗时的中位数/p95、工作块数，以及逐轮四段之和、整链首尾跨度和间隔。不能把四个算子各自的中位数直接相加当作整批中位数，也不能把API全程初始化/分配/释放平均时间当作稳态提交时间。

采集有额外成本，本次主机和设备计时只用于诊断，不加入先前6轮无采集样本，也不拿旧版采集和新版无采集计时直接相减宣称算法收益。

## 本次代码

只更新`standalone/analyze_pipeline_profile.py`，没有改动算子、C++调用器、CPU或样本，无需重新编译。

`analyze`新增识别`document_sharded_exact_radix_topk_v1`，检查零额外GM候选空间及同版本安装记录中的已验收工作区引用。Radix工作区按自身独立验收值检查，不套旧版或Warp的候选空间公式，不假定它等于贡献算子的工作区。

CSV表头/单位、设备号、stream、四段输入形状、任务顺序和数量、op_summary与task_time对应任务及时间、六数组指纹、首末输出的原检查继续保留。源表和运行文件只读；表格不完整或规则不符就停止，不猜测正式轮次或补造数据。

本机19项Radix分析适配检查、原四段22种失败场景和Warp回归通过，使用合成表，不是本次Radix实际设备分段结果。

## 黄区执行

提交同步代码后逐行执行。命令沿用本机CANN环境此前已核对并成功采集旧版/Warp的选项，目录独立，不重跑普通6轮基准：

```bash
cd /test/w50066029/bm25-npu
source /usr/local/Ascend/cann-9.2.0/set_env.sh
python3 verify_package.py
set -o pipefail
msprof --output=results/msprof_radix_64k_q1_s32_1 --runtime-api=on --task-time=l1 --ai-core=on --aic-metrics=PipeUtilization --python-path=/usr/bin/python3 python3 standalone/run_sharded_pipeline.py --input results/normal_64k_q1 --output results/radix_64k_q1_s32_msprof1 --runner results/radix_sharded_host_build/bm25_sharded_benchmark --vendor-dir results/radix_sharded_runtime/vendors --shards 32 --device 2 --warmup 5 --repeats 10 --mode resident 2>&1 | tee results/msprof_radix_64k_q1_s32_1.log
```

实际四段执行与结果检查成功、工具结束后，再分析：

```bash
python3 standalone/analyze_pipeline_profile.py --profile-root results/msprof_radix_64k_q1_s32_1 --run results/radix_64k_q1_s32_msprof1 --output results/radix_64k_q1_s32_profile_analysis1.json 2>&1 | tee results/radix_64k_q1_s32_profile_analysis1.log
```

出现缺表、表头变化或任务数量不符时，先保留日志定位，不放宽检查来凑结果。若性能工具环境发生变化，以实际错误和帮助为准。可选择空闲的同规格卡，避开设备4；同时改变采集运行的device参数即可，分析器从实际元信息核对设备。

黄区文件留黄区，只提供允许分享的终端分析摘要，不打包或传出源码、表格。

## 拿到结果后如何选择

若Radix局部Top-K仍占大头，才进一步分析其重复扫描、计数或同步；仅总耗时不能区分这几种原因。若累加或其他阶段占比上升，则按新的实测瓶颈安排工作。采集到的指令比例和工作块数不等于全卡利用率。

随后用相同输入和计时边界做旧版、Warp、Radix的同期比较，再安排CPU单查询Top-K合理并行、BM25S独立对手、规模/批量曲线和含传输到完整查询验收。不因Radix候选更少就直接认定算法更快，也不因当前多数轮胜出就结束整体任务。

建议commit备注：`feat: 支持Radix四段设备性能采集分析`
