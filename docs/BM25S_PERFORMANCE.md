# 原版 BM25S 对照与后续真实数据测试

当前验收目标按用户最新要求调整为：NPU 性能与整改前接近，CPU 主基准使用原版 BM25S；先完成一轮同规模合成数据回归，确认后再使用真实数据集。C++ 新旧配对保留作辅助诊断，不作为这轮的 CPU 主对手。

## 本轮测量口径

- 复用已有千万篇合成输入、已在 A5 验收的主机二进制和 vendor，不重建算子。Q8、每查询 3 词、K10、FP32；完整 Search 到 CPU 收到 Top-K，建库、首次上传、JIT 和预热另计。
- `--cpu-baseline bm25s` 使用固定提交 `03eac5a8fb3aa4958ce37415a4cf2220d35157f7` 的未修改原版源码，调用其原生 `retrieve`。每轮测 1/8/32 线程，并与同轮较快配置比较。严格同分适配器不计入这一原生基准。
- 共六轮，NPU 与三个 BM25S 配置按正反顺序交替；每进程 5 个预热周期、A/B 各 30 个计时样本。保留原有分数、索引和重复输出检查。原版 BM25S 的同分 ID 顺序与项目严格合同可能不同，原生耗时胜出不能冒充严格同分合同等价；NPU 自身仍接受独立严格 oracle 检查。
- 报告包含每轮中位批延迟、P95、Search 时间吞吐 QPS及原生 BM25S 倍率。另将 NPU 与 2026-10-29 同规模、同生成条件和查询的 BM25S 会话历史记录比较，打印 `Historical NPU comparison`。历史参考是另一会话，不能冒充同期旧版配对。
- 历史千万篇 NPU A/B 中位批延迟约为 2.003/1.978 ms，原版 BM25S 约 38.256/38.271 ms。这些仅为历史参考；新会话倍率和是否保持速度都以新测量为准。
- 用户的目标是与此前性能接近，尚未指定固定百分比阈值。报告输出变化比例和波动，保持 `performance_acceptance: pending_review`。若低干扰重复对照仍稳定慢 4%～5%，必须记录为实测开销，不能继续笼统归因于系统负载。

## A5 执行

先在执行窗口实际预留 CPU 0～31 对应的物理核及其 SMT 兄弟线程（本机为 128～159），内存使用节点 0。下面的 `numactl` 负责放置，不会自动预留资源。复用原项目的私有 Numba 依赖，校验失败会停止；不自动升级系统包。

```bash
cd /test/w50066029/bm25-npu-indexsdk
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.14.1/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
python3 -c 'from pathlib import Path;from standalone.prepare_bm25s_numba import installed_record;installed_record(Path("/test/w50066029/bm25-npu/results/bm25s_numba_py39"));print("BM25S private dependencies verified")'
numactl --physcpubind=0-31 --membind=0 python3 -u standalone/benchmark_final.py --cpu-baseline bm25s --input /test/w50066029/bm25-npu/results/large_scale_close1/inputs/10000000 --output results/bm25s_regression_10m_01 --vendors /test/w50066029/bm25-npu/results/final_runtime/vendors --host-build results/indexsdk_a5_acceptance_02/build --dependency-dir /test/w50066029/bm25-npu/results/bm25s_numba_py39 --device 5
```

私有依赖目录若不存在，在本项目新目录离线安装：`python3 standalone/prepare_bm25s_numba.py install --output results/bm25s_numba_py39`，并将上面校验及运行命令中的依赖目录改为该绝对路径。已有目录不可覆盖，失败时保留日志。

输出目录必须是新的。原版 BM25S 本轮会单独建库并记录构建耗时；该时间不计入 Search。最后回传 `Historical NPU comparison`、`Final Search decision`，以及 `results/bm25s_regression_10m_01/report.json`。程序完成表示正确性及测量检查完成，不自动批准性能差异。

## 下一步真实数据集

合成数据回归通过后，再确定真实数据集的路径/格式、文档与查询范围、统一分词和词表、K/批量以及是否有相关性标注。BM25S 与 NPU 必须使用同一份文档、查询和评分参数；原生同分行为单独记录。

当前 `large_corpus.py --corpus-jsonl` 入口接收每行一个词 ID 数组；本轮基准固定 Q8/3 词/K10，不能直接宣称覆盖可变长度真实查询。收到数据格式后再准备对应转换及查询容量，分别测建库、完整检索延迟/吞吐和正确性；有标注时再测检索质量。合成数据的速度和倍率不预先扩展到真实数据。
