# CPU常驻索引Search：补齐与NPU相同的查询边界

## 已确认的NPU实机进展

用户黄区设备2日志确认，新原始词编号Search程序已编译、公共输入测试通过、六个实际API函数来源核对通过，随后A/B/A/全空/B五次实际Search和全部结果检查通过。完整分数、局部候选、严格排名、重复输出和静态索引复用均通过。

| 调用 | 查询数 | 实际记录的整批诊断时间(ms) |
|---|---|---|
| call0_A | 6 | 0.144336 |
| call1_B | 6 | 0.066530 |
| call2_A | 6 | 0.078357 |
| call3_empty | 3 | 0.092979 |
| call4_B | 6 | 0.066771 |

这些时间包含请求检查/整理、上传、接口准备、三段计算、同步、CPU取得Top-K及必要检查/请求资源释放。初始化和调试下载单列。**这是五次含调试间隔的单次诊断，不是稳态统计，不能把6查询整批时间当作Q=1延迟，也不能与旧CPU展开数组时间相除。** 该日志没有展示冷启动具体数值，以黄区报告字段为准。

用户后续黄区日志确认：**新CPU程序已编译，180次独立FP32公式/严格排名/交替调用、16项错误/失败停止检查，以及小语料五次CPU/NPU直接交叉全部通过。** CPU五次整批诊断为0.007261、0.006700、0.006200、0.005048、0.006239ms，非稳态。下一步见[正常规模Search验收步骤](正常规模Search验收步骤.md)。下文编译命令保留为复现记录。 现有NPU算子、评分/三段/Search程序、共享词编号头文件和已有CPU Top-K源码原字节均不改，因此不用重编或重装NPU。

## 简单说明：输入、处理、输出、验收

输入与NPU相同：提前建好的整库原始TF倒排索引，以及CPU内存中按查询分组的原始词编号。CPU也允许初始化时把索引读入内存，并声明查询容量。

`CpuSearchSession::Search`从入口第一行开始计时，依次做：

1. 检查本批查询数量、词编号和容量；保留重复词，未知词在查询计算时跳过，空查询保留。
2. 将“查询＋一段文章范围”交给CPU线程。每个线程先清零自己负责的文章分数，再按原查询词顺序，在对应词的有序倒排记录中二分定位该文章段，只处理命中记录。
3. 使用与NPU相同的FP32 TF、IDF、长度项及公式，按相同查询词顺序累加。不同文章范围不重叠，不用浮点原子加法。
4. 复用已验收的`TopKDocumentShards`：各段用最多32个候选的小堆，再合并出最终Top-K。文章分区允许不整除，局部文章少于K也可处理；最终编号仍是全库编号。
5. CPU已有FP32分数及INT32文章编号，完成必要范围/严格顺序检查后停表。

CPU直接使用原始二维词编号，不需要上传或NPU后端的连续数组，因此不为了形式相同而加入无用的打包和传输。双方比较的是**同样的逻辑输入到同样的CPU可用输出**，各自必需处理都计时。文件解析、索引加载/检查、通用容量分配、OpenMP团队启动及全空容量预启动单列。调试污染、完整总分导出、独立公式核对和索引源文件重读均在Search外。

输出是五次真正CPU Search返回的Top-K、核对用完整文章分数、实际日志、单次时间和报告。CPU C++程序不读期望答案，也不读旧的逐查询展开TF/offset文件。CPU全分数必须与同输入有序FP32参考逐字节一致，Top-K编号按分数降序/同分编号升序，选中分数保留原字节；同时按原容差核对独立原始语料公式。

指定`--npu-reference`时，只检查**已经存在的NPU实际运行报告与文件**：索引/查询/K、编译源码、程序/日志/输出指纹及原始词编号文件必须对应。然后CPU/NPU每条查询的编号和顺序严格一致，完整分数及Top-K分数在原容差内一致。这不是再次执行NPU，也不把CPU结果写到NPU目录。

CPU线程数、文章分块数可调，使用`-O3 -march=native -ffp-contract=off -fno-fast-math`和OpenMP；不是慢速Python循环。首次4线程、2个文章分块只用于6篇小语料正确性；后续正常规模必须检查合理线程/分块配置，不能称4或8线程为整机最优CPU，也不能因为NPU更快就挑一个慢CPU配置。

## 黄区现在执行

提交推送后拉取。只编译新的CPU工程，不需要CANN、BM25S、Faiss或IndexSDK运行依赖。命令逐条成功再继续，失败就停：

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
set -o pipefail
cmake -S cpu_reference/index_search -B results/cpu_index_search_build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON 2>&1 | tee results/cpu_index_search_configure1.log
cmake --build results/cpu_index_search_build -j 2>&1 | tee results/cpu_index_search_build1.log
results/cpu_index_search_build/bm25_cpu_index_search_tests 2>&1 | tee results/cpu_index_search_tests1.log
```

C++单元测试源码覆盖1/6/129/2048篇文章、K=1/3/10/32及全部文章的适用组合，1/4/8线程、不同分区、空/重复/未知词和交替查询，并从原始文章独立统计TF/DF生成FP32参考；还检查错误输入及失败后不能继续调用。**用户已实际运行通过180次查询和16项错误/会话停止检查；这些是黄区真实C++执行结果，区别于本机模拟测试。**

编译和单元测试通过后，使用原小语料，在同一进程执行五次CPU Search，并直接对照已经通过的NPU Search目录：

```bash
python3 standalone/run_cpu_index_search.py --input results/query_score_small_input1 --output results/cpu_index_search_small_t4_s2_check1 --threads 4 --document-shards 2 --k 3 --max-queries 65 --max-words 512 --npu-reference results/search_small_s2_k3_check1 2>&1 | tee results/cpu_index_search_small_t4_s2_check1.log
```

如果之前通过的新NPU Search输出目录叫别的名字，改`--npu-reference`，不能用旧三段诊断或旧resident结果代替。CPU目录已存在就改check2，不覆盖旧报告。若NPU参考可执行程序或源码已改变，先定位来源，不跳过检查。本次不改这些文件，因此现有NPU报告可复用。

成功状态为`cpu_index_search_and_independent_checks_passed`，五项`npu_cross_checks`也应全部通过。只贴允许分享的终端汇总或首个错误，黄区文件留在黄区。

## 关键函数怎么向MT讲

- `Index`：一次读取五组原始TF索引，检查类型尺寸、有序倒排及有效统计量；查询只读这份主机内存。最后额外重读原文件核对字节，调试重读不算第二次生产索引加载。
- `CpuSearchSession::Search`：从原始词编号开始计时，完成索引范围查找、FP32计算和Top-K；每个文章位置每次都清零/覆盖，没有预存查询答案。
- `TopKDocumentShards`：复用现有小堆/合并实现，没有重新发明选择算法。完整总分导出只给核对，不包含在正式返回值里。
- `check_cpu_outputs/compare_npu`：分别核对CPU完整FP32公式与严格排名、CPU/NPU结果直接一致；不比较前后两次日志里的单次耗时。
- `validate_npu_reference`：确认拿来核对的是同一数据和规则的已验收新Search，而不是旧计算实验。无需加载或执行CANN。

本机67项协议/计时记录/严格交叉/错误处理检查使用假CPU/NPU可执行文件、模拟CPU子进程及模拟NPU历史报告；这些临时文件已删除，**不是实际CPU或NPU执行证据**。

两个公共入口现已在小语料跑通，接着扩展到正常64K/Q1和256K/Q1、Q8，补原版BM25S的计时边界与Numba参考，再进行预热、多轮交替CPU/NPU、中位数/p95及批量吞吐报告。现有五次CPU和NPU诊断不直接计算稳态加速比，最终仍按[search合同](search计时与验收合同.md)。

建议commit备注：`feat: 增加同边界CPU常驻索引Search与NPU结果交叉核对`
