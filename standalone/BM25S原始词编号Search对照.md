# 原版BM25S Numba：原始词编号到Top-K的Search对照

## 先说明现在到哪里了

用户已在设备4完成新版NPU与32线程C++ CPU的同期完整Search六轮：A CPU0.0742165ms/NPU0.332506ms，B CPU0.07401325ms/NPU0.3325485ms，均为每轮中位数再取中位数。A/B的NPU都快0/6轮，耗时约CPU的4.48/4.49倍；全部结果核对通过，性能目标未达成。这是262144篇文章、8条查询、K=10的整批时间，不是单条延迟，不代表950硬件的一般能力。

本轮补齐成熟CPU对手：固定提交`03eac5a8fb3aa4958ce37415a4cf2220d35157f7`的原版BM25S，使用黄区已私有安装的Numba0.60/llvmlite0.43及已有NumPy2.0.2。原版25个文件和原参考包装程序都不改，不重装或升级系统依赖，不调用NPU。

## 输入、步骤、输出及验收

输入直接复用`results/search_256k_q8_input1`的原始文章和A/B/空查询词编号，不改用旧的展开TF输入。索引在测量前建好并常驻CPU内存；BM25S提前存各词对各文章的贡献，当前NPU保留原始TF、查询时算公式，因此同边界耗时也不等于纯硬件比较。

1. 核对原版25文件指纹和实际加载位置；只在本Python进程加入已安装私有Numba目录，验证安装记录、版本、实际模块位置及JIT签名。Numba失败即停，不退回NumPy。
2. 从同一批原始文章建立原版BM25S索引，采用ATIRE词频项、Lucene IDF、k1=1.2、b=0.75、FP32，保持原公式。词表ID仍对应原始整数词编号，另加一个真实零列处理空查询。
3. 核对A/B/空查询的全部文章分数，使用原独立公式、原容差和有序FP32参考；严格适配编号必须与原参考完全一致，不按误差容差合并同分。首次评分JIT、首次原版串行/并行检索JIT、建索引时间单列。
4. 对每个线程配置分别测两条路线。每条路线先验证A/B/A/空/B和重复字节，再做5个A/B预热周期、30个A/B正式周期，周期内交替AB/BA。完整公式、原始分数字节、排序和重复检查在正式计时外进行。

两条路线分别为：

| 路线 | 实际计算与计时内处理 | 结果规则 |
|---|---|---|
| `original_native` | 原版`retrieve`及其原版Numba串行/并行实现；包装器只补原始首条空查询的入口兼容，原版处理未知词、数组打包与检索 | 核对原版自身前K大分数及原始FP32位；同分编号差异单独保存，不能当作满足严格合同 |
| `strict_adapter` | 计时内过滤未知词、补零列；调用原版编译评分，再用已有NumPy阈值选择/同分编号补齐/只排入选K项，多个查询使用本次调用创建的线程池 | 分数降序、同分原文章编号升序；这是独立适配版，不能冒称原版Numba检索或全局最优严格实现 |

原版`bm25s/__init__.py::is_list_of_list_of_type`只看第一条查询的第一个词，首条为空时会先拒绝。包装器在`SearchSession.search`内把空行换为已索引的零列占位，再调用原版。未知词、重复词仍按原合同处理；严格适配的相应整理也在Search内。没有把预过滤后的查询提前准备好再开始计时。

计时边界是：CPU已有原始分组整数词编号 → 计时开始 → 请求类型/容量检查、必要空查询兼容/未知词处理、数组转换、索引检索、Top-K、必要输出检查及本次请求临时对象释放 → CPU持有Top-K → 计时停止。文本分词不涉及本数字语料；建索引/贡献预计算、首次JIT、预热、调试分数准备、文件IO和独立正确性核对单列或排除。原生路线保留自身同分规则，严格合同只由严格适配路线承担。

输出保存所有A/B样本、中位数、p95、吞吐、原版同分差异、首次调用/JIT时间、原版编译签名、依赖/索引/输入/输出/源码指纹。p95采用与C++对照相同的最近秩定义。1和8是请求的查询工作线程数，Numba原版并行的是查询维度；不等于占用相同数量的物理核，也不同于C++的32个文章分区。严格适配还受原版评分编译方式及Python线程池开销影响，分别报告，不声称已穷尽优化。

本轮是同一个模型/进程内的重复调用，先验收目标Numba和计时入口；不是独立进程六轮，不与历史NPU时间直接拼正式加速比。目标环境通过后，才选更合适配置做后续同期比较或定位新NPU组合瓶颈。

## 提交与执行

Windows提交本轮新增文件：

```powershell
cd F:\Ascend\bm25\bm25-a5
git add -- standalone/run_bm25s_index_search.py standalone/test_bm25s_index_search.py package_manifest.json README.md 'standalone/*.md'
git commit -m "perf: 增加原版BM25S Numba的原始词编号Search计时"
git push
```

黄区复用现有私有依赖，不重新下载或安装。无需CANN编译或source新的算子环境：

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
set -o pipefail
```

```bash
python3 standalone/run_bm25s_index_search.py --input results/search_256k_q8_input1 --output results/bm25s_numba_search_256k_q8_ref1 --dependency-dir results/bm25s_numba_py39 --workers 1 8 --warmup 5 --repeats 30 2>&1 | tee results/bm25s_numba_search_256k_q8_ref1.log
```

完整稳态样本在首次编译和预热之后；看到`compile_and_check_raw_ID_search`时可能正在首次编译，不能把这段等待当成每次查询耗时。失败停在第一处，保留日志；结果目录存在则换名字，不覆盖，不改原版解决差异。黄区文件留黄区，仅分享允许的错误或末尾汇总。

成功状态应为`original_bm25s_numba_raw_ID_search_and_separate_strict_adapter_passed`。原生同分不匹配行数可以非零，这是已声明的原版规则差异；严格适配不允许编号或顺序不同。数值误差不允许放宽原合同。

## 对应到关键函数和当前完成程度

`SearchSession.search`只持有静态模型和配置，接收每次新的原始查询，在方法内完成验证/兼容/实际检索和必要输出检查；它不持有期望答案或查询结果缓存。`timed_search`在调用此方法之前启动时钟、返回之后停止。`compiled_signatures`核对原版评分及串行/并行检索确实生成nopython签名。`run`建索引、准备独立核对、区分冷启动与正式A/B样本，并原样记录原生同分差异。

本地74项使用明确假模型的空/未知/重复词、容量、非法输入、计时位置和输出错误检查通过。另在本机Python3.10、NumPy2.2.6、Numba0.68上实际跑通固定原版的串行及并行检索、严格适配、全分数/排名/重复/空查询核对；这是接口集成，不是黄区Python3.9/Numba0.60兼容性或性能结果。目标环境运行仍待用户执行上述命令。
