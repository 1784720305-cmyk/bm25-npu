# 原版 BM25S 对照与后续真实数据测试

当前验收目标按用户最新要求调整为：NPU 性能与整改前接近，CPU 主基准使用原版 BM25S；先完成一轮同规模合成数据回归，确认后再使用真实数据集。C++ 新旧配对保留作辅助诊断，不作为这轮的 CPU 主对手。

## 本轮测量口径

- 复用已有千万篇合成输入、已在 A5 验收的主机二进制和 vendor，不重建算子。Q8、每查询 3 词、K10、FP32；完整 Search 到 CPU 收到 Top-K，建库、首次上传、JIT 和预热另计。
- `--cpu-baseline bm25s` 使用固定提交 `03eac5a8fb3aa4958ce37415a4cf2220d35157f7` 的未修改原版源码，调用其原生 `retrieve`。每轮测 1/8/32 线程，并与同轮较快配置比较。严格同分适配器不计入这一原生基准。
- 共六轮，NPU 与三个 BM25S 配置按正反顺序交替；每进程 5 个预热周期、A/B 各 30 个计时样本。保留原有分数、索引和重复输出检查。原版 BM25S 的同分 ID 顺序与项目严格合同可能不同，原生耗时胜出不能冒充严格同分合同等价；NPU 自身仍接受独立严格 oracle 检查。
- 报告包含每轮中位批延迟、P95、Search 时间吞吐 QPS及原生 BM25S 倍率。另将 NPU 与 2026-10-29 同规模、同生成条件和查询的 BM25S 会话历史记录比较，打印 `Historical NPU comparison`。历史参考是另一会话，不能冒充同期旧版配对。
- 历史千万篇 NPU A/B 中位批延迟约为 2.003/1.978 ms，原版 BM25S 约 38.256/38.271 ms。这些仅为历史参考；新会话倍率和是否保持速度都以新测量为准。
- 用户的目标是与此前性能接近，尚未指定固定百分比阈值。报告输出变化比例和波动，保持 `performance_acceptance: pending_review`。若低干扰重复对照仍稳定慢 4%～5%，必须记录为实测开销，不能继续笼统归因于系统负载。

## 2026-11-05 回归结果

用户回传了 `bm25s_regression_10m_01` 的全部 24 份进程摘要及最终判定，已在本地重算六轮顺序、1/8/32 线程较快者、历史差值与 QPS，记录见 [本轮证据](../evidence/bm25s_regression_a5_20261105.json)。原始逐样本文件和主报告未单独下载。

| 每批 Q8 的汇总 | A | B |
|---|---:|---:|
| NPU 中位延迟 ms | 1.992866 | 1.978685 |
| NPU P95 ms | 2.000526 | 2.001111 |
| 原版 BM25S 同轮较快者中位延迟 ms | 31.382953 | 31.445556 |
| CPU/NPU 延迟比 | 15.7476 | 15.8921 |
| NPU 相对历史中位延迟变化 | -0.5075% | +0.0443% |

NPU Search 时间吞吐约 4045/4046 QPS，A/B 均六轮领先，P95 也逐轮低于三个原生 CPU 配置。原生同分语义不等同于项目严格合同，这仍是原生对照结论。倍率由历史约 19 倍变为约 15.8 倍，是本会话 BM25S 较历史快约 18%，没有观察到 NPU 数倍降速。

按用户“与此前性能接近”的目标，本轮合成数据回归满足要求，可以进入真实数据阶段。该判断限于已回传的千万篇/Q8/K10 工作负载，不是统计等价性证明，也不解决独立的 C++ 参数开销问题。

## 大页分配修复后的 NPU 回归

`d685cf4` 修复大块设备内存分配后，`results/indexsdk_alloc_host_01` 在 A5 完成全套 host/device 验收。原千万篇、Q8/3 词/K10 输入的 NPU 六轮中位汇总为 **A 1.7590635 ms、B 1.74224325 ms**；六轮 P95 的中位数为 **1.7642815/1.767821 ms**。相对此前本页的 1.99286625/1.97868525 ms 分别低 **11.73%/11.95%**，当前未见回退。

该回归包含验收用 C++ CPU 门槛，不能据此生成新的合成库原版 BM25S 倍率。只比较相同 NPU 输入和计时边界；不同会话的 CPU 角色及进程顺序不同，也不能据此独立量化大页策略的收益。[本轮证据](../evidence/msmarco_a5_20261105.json)保存全部六轮 NPU 摘要、执行指纹及对比计算。

真实 [MS MARCO 884 万篇](MSMARCO_REAL_DATA.md#2026-11-05-a5-实测结果)也已完成：原版 BM25S Q8 A/B 为 24.856/24.032 ms，NPU 为 3.124/3.119 ms，延迟比 7.957/7.704；两组中位与 P95 均六轮领先。真实语料的查询、词表与词频分布不同，不能把该绝对时间直接当作合成库回退。

## A5 复现

上述结果已完成，无需为归档重复测量；如复现，使用新的输出目录。

先在执行窗口实际预留 CPU 0～31 对应的物理核及其 SMT 兄弟线程（本机为 128～159），内存使用节点 0。下面的 `numactl` 负责放置，不会自动预留资源。复用原项目的私有 Numba 依赖，校验失败会停止；不自动升级系统包。

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
python3 -c 'from pathlib import Path;from standalone.prepare_bm25s_numba import installed_record;installed_record(Path("/test/w50066029/bm25-npu/results/bm25s_numba_py39"));print("BM25S private dependencies verified")'
numactl --physcpubind=0-31 --membind=0 python3 -u standalone/benchmark_final.py --cpu-baseline bm25s --input /test/w50066029/bm25-npu/results/large_scale_close1/inputs/10000000 --output results/bm25s_regression_10m_02 --vendors /test/w50066029/bm25-npu/results/final_runtime/vendors --host-build results/indexsdk_alloc_host_01/build --dependency-dir /test/w50066029/bm25-npu/results/bm25s_numba_py39 --device 5
```

私有依赖目录若不存在，在本项目新目录离线安装：`python3 standalone/prepare_bm25s_numba.py install --output results/bm25s_numba_py39`，并将上面校验及运行命令中的依赖目录改为该绝对路径。已有目录不可覆盖，失败时保留日志。

输出目录必须是新的；示例使用当前已经验收的主机程序，已有测试无需为归档重复测量。原版 BM25S 会单独建库并记录构建耗时；该时间不计入 Search。复现时回传 `Historical NPU comparison`、`Final Search decision`，以及新结果目录的 `report.json`。程序完成表示正确性及测量检查完成，不自动批准性能差异。

## 下一步真实数据集

用户暂无真实数据，选择先用容易复现的公开数据集。已选 BEIR SciFact，并提供 [真实数据执行流程](SCIFACT_REAL_DATA.md)：5183 篇文档、300 条测试查询、339 条相关性标注；全量质量验证与 Q8 性能测量分开。

新真实数据入口支持自然变长查询，按实际总词数声明容量；原合成数据默认仍保持 Q8/3 词/K10。SciFact A5 首轮已完成：全量质量指标对齐，Q8 样本 A 慢 3.37%、B 快 5.37%，小库性能结论为有胜有负；完整数值见上述真实数据文档，合成数据倍率不扩展到该数据。
