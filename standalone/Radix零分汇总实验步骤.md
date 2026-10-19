# Radix零分汇总：独立实验及验收

最新实机状态（2026-10-19）：本页算子编译/安装、23组115次独立设备验收、四段小样本及256K/8查询、CPU交叉均已由用户通过。新版大样本0.417896ms仅为3次测量诊断，不证明性能提高。下一步直接按 [新旧同期对比步骤](Radix新旧同期对比步骤.md) 执行，不重复生成、编译、安装或23组验收。以下命令保留供复现，已存在的输出目录勿重复执行或删除。

## 为什么做这一步

当前262144篇文章、8条查询、K=10的两组六轮驻留计算均正确。与按查询并行CPU相比，NPU六轮更快；与更快的CPU文章分8份版本相比，六轮中位数再取中位数为CPU 0.4169955ms、NPU 0.413235ms，基本持平，尚无有说服力的稳定优势。CPU线程数为8，不等于整机CPU最佳性能。

用户本轮实际设备2采集有效：68条BM25任务交叉核对，排除首次1轮、检查预热5轮、普通预热1轮，统计正式10轮。贡献6.413us、累加86.572us、Radix局部Top-K301.586us、合并5.145us；逐轮四段设备耗时之和中位数399.734us，链跨度400.589us，链内间隔之和0.066us。这是采集诊断，不把四个阶段各自中位数相加当作整批中位数。

本例2097152组中1910309组没有贡献，约91%。当前累加仍给空组写零，Top-K仍扫描完整分数。源码事实是：原Radix按11/11/10位进行三轮计数，每个满足当前前缀的分数都对共享直方图做一次整数atomicAdd。数值为零的分数具有同一个key，会更新同一个桶；门槛落在零附近时，后续轮次仍可能计数这些零。

推断：大量零集中更新一个桶，可能带来开销。但301.586us中到底多少是计数争用、多少是候选收集的同步或其他指令，尚未直接测出。本实验只验证减少零分atomic更新这一项，不提前认定必然变快。

## 输入、处理、输出

输入与原版相同：二维FP32分数数组和K，有限非负分数，允许负零，K在1..min(D,32)。完整四段输入、贡献公式、CPU程序均不变，不预计算BM25贡献，也不从CPU传入答案。

处理过程：

1. 每个线程扫描自己负责的分数。正分仍按原方式更新共享桶；正零和负零先计入这个线程自己的整数计数。
2. 用原来验证非法输入时已经调用的PrefixPair，把128个线程的零数量汇总。非法输入检查继续保留。
3. 在读取直方图时，将零总数计入零对应的最后一个桶。两个读取位置（线程负责的桶区间总数、桶内寻找第K名）都使用这个逻辑计数。不增加一次共享写或一次atomic来补零数。
4. 后续选择门槛、按文章编号收集候选、最后排序选出的K项，保持原代码。零仍参与选择；不能因为正分不足K就丢掉零分结果。

零的比较key为UINT32_MAX，所以三轮分别对应桶2047、2047、1023。零计数只在它匹配当前轮前缀时累加；若上一轮已经选中了不含零的前缀，本轮零总数就是0。正的次正规数按原始位判断为非零，不能当作零丢掉。输出重新读取原分数字节，因此正负零的输出符号也保留。

输出仍为FP32 Top-K分数、INT32文章编号。分数降序、数值同分文章编号升序；公式容差不变，排名和选中分数字节仍严格核对。

代价也要测：新版本增加线程私有零计数、归约的有效数据和桶读取时的加法/判断。虽然源代码未增加PrefixPair调用次数、屏障次数或分数扫描轮数，原版第二归约通道是常量0，编译器可能曾省掉其中部分指令。因此“没有新增函数调用”不等于“没有新增指令”。非零较多的输入可能不受益，甚至退化。

## 代码与隔离

| 文件或目录 | 作用 |
|---|---|
| npu_prototype/topk_radix_zero/op_kernel/bm25_top_k.cpp | RadixTopKSimt仅改变三轮直方图零计数。PrefixPair、ValidBits、ScoreKey及候选收集至写出部分与原版相同 |
| npu_prototype/topk_radix_zero/op_host | 同样的IR、形状、分核、8512字节算法UB及系统工作区规则 |
| npu_prototype/topk_radix_zero_host/CMakeLists.txt | 针对新安装头文件构建独立诊断程序；复用原诊断C++源文件，不复制或修改它们 |
| standalone/validate_topk_radix_zero.py | 新工程/vendor/源码/设备目标文件来源核对、交替输入及逐次完整排序验收；独立版本标识 |
| standalone/check_radix_zero_model.py | Python整数及线程模型，逐轮全部桶计数、新旧Radix、独立完整排序和原FP32字节核对 |
| standalone/prepare_sharded_runtime.py | 完整独立验收通过后才能建radix_zero运行目录；不能借用旧版19组报告 |

新工程results/Bm25TopKRadixZero，安装vendor为bm25_topk_radix_zero。旧版topk_radix、独立验收脚本、诊断C++/CMake、CPU及二进制样本均保留原字节；因此旧版验收引用仍可复核。四段C++未改，可复用已通过的results/radix_sharded_host_build/bm25_sharded_benchmark，不需要重编它。

四段版本标识为document_sharded_exact_radix_zero_aggregate_topk_v1；原版仍为document_sharded_exact_radix_topk_v1。C++计时里的topk_implementation=radix表示同一接口/工作区家族，不能单靠这一字段区分新旧，须看run_metadata.json的topk_variant、安装路径及源码指纹。新旧运行目录必须分别建立。

当前CANN编译、安装、上述独立及串联设备正确性已由用户通过，性能收益仍未验证。实际状态以页首及对应报告为准；模型检查不能替代设备验收。

## 第一步：只生成并编译新算子

同步代码后在黄区逐行执行。任何一步失败就停止，不删除或覆盖旧工程。

```bash
cd /test/w50066029/bm25-npu
source /usr/local/Ascend/cann-9.2.0/set_env.sh
python3 verify_package.py
set -o pipefail
python3 standalone/create_project.py --operator topk_radix_zero --compute-resource ai_core-ascend950 --compute-unit ascend950 --msopgen-python python3
cd /test/w50066029/bm25-npu/results/Bm25TopKRadixZero
bash build.sh 2>&1 | tee ../radix_zero_build1.log
```

build_out/custom_opp_openEuler_x86_64.run生成成功才进入下一步；编译成功不等于设备运行正确。

## 第二步：独立安装并读实际接口

```bash
bash build_out/custom_opp_openEuler_x86_64.run --install-path=/test/w50066029/bm25-npu/results/custom_opp
cd /test/w50066029/bm25-npu
cat results/custom_opp/vendors/bm25_topk_radix_zero/op_api/include/aclnn_bm25_top_k.h
ldd -r results/custom_opp/vendors/bm25_topk_radix_zero/op_api/lib/libcust_opapi.so > results/radix_zero_dependencies1.log 2>&1
cat results/radix_zero_dependencies1.log
```

核对新头文件的实际两阶段声明及依赖。虽然IR与旧版相同，仍以实际文件为准；脚本会严格比较此前实际确认的签名，不匹配就停止。无需把新旧多个Top-K vendor一起source。Python验收器会只给子进程设置本次vendor，调用器按绝对路径加载并检查符号归属。不经过ATC或OM；原ATC运行库问题未因此认定解决。

## 第三步：编译独立诊断调用器，先冒烟再全套

实际接口和依赖无异常后执行：

```bash
cmake -S npu_prototype/topk_radix_zero_host -B results/topk_radix_zero_host_build -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 -DBM25_OPP_ROOT=/test/w50066029/bm25-npu/results/custom_opp/vendors/bm25_topk_radix_zero 2>&1 | tee results/radix_zero_host_configure1.log
cmake --build results/topk_radix_zero_host_build -j 2>&1 | tee results/radix_zero_host_build1.log
python3 standalone/validate_topk_radix_zero.py --suite smoke --output results/radix_zero_topk_smoke1 --device 2 2>&1 | tee results/radix_zero_topk_smoke1.log
```

冒烟两组通过后运行全套：

```bash
python3 standalone/validate_topk_radix_zero.py --suite all --output results/radix_zero_topk_validation1 --device 2 2>&1 | tee results/radix_zero_topk_validation1.log
```

全套23组保留原19种尺寸，追加8192列K=10/32、大尺寸全零、257行跨分核等。行类型覆盖正负零、全同分、随机、无零、次正规数、最大有限数、恰好K-1/K/K+1个正分及约91%零。每组默认同进程A/B/A/B/A五次，先填无效输出，每次C++完整排序比较全部编号及原FP32位；Python另核对两套输出。失败、中断、缺文件、尺寸/来源/工作区不符都停止，不用CPU答案替代。冒烟不够，旧验收报告也不能认证新版。

## 第四步：全套通过后，接原四段小样本

```bash
python3 standalone/prepare_sharded_runtime.py --topk-implementation radix_zero --output results/radix_zero_sharded_runtime
python3 standalone/run_sharded_pipeline.py --input results/pipeline_input --output results/radix_zero_small_s2_check1 --runner results/radix_sharded_host_build/bm25_sharded_benchmark --vendor-dir results/radix_zero_sharded_runtime/vendors --shards 2 --device 2 --warmup 2 --repeats 3 2>&1 | tee results/radix_zero_small_s2_check1.log
```

小样本全部六数组、首末/重复字节检查通过后，检查实际目标输入：

```bash
python3 standalone/run_sharded_pipeline.py --input results/normal_256k_q8_batch1 --output results/radix_zero_256k_q8_s32_check1 --runner results/radix_sharded_host_build/bm25_sharded_benchmark --vendor-dir results/radix_zero_sharded_runtime/vendors --shards 32 --device 2 --warmup 2 --repeats 3 2>&1 | tee results/radix_zero_256k_q8_s32_check1.log
python3 standalone/compare_compute_outputs.py --input results/normal_256k_q8_batch1 --cpu results/cpu_256k_q8_scan1/r1_s8 --npu results/radix_zero_256k_q8_s32_check1 --report results/radix_zero_256k_q8_cross1.json 2>&1 | tee results/radix_zero_256k_q8_cross1.log
```

不能用这三次诊断时间作最终性能结论。原有小样本和大规模二进制输入不重新生成。

## 正确后如何决定保留或放弃

在同一卡、同输入、同S=32、同预热/重复和计时边界下交替测试旧Radix与新版。全部结果必须正确，先比较整条四段resident路径，再用采集比较Top-K设备段；采集不混入正式基准。测试原64K/单查询、256K/单查询、256K/8查询，补全零、少量正分和非零较多的独立Top-K输入。当前独立诊断程序不提供性能成绩，不能把其每次poison/上传/下载/完整排序耗时当内核时间；如需独立微基准另建明确计时入口。

若减少atomic次数后整链无可重复收益，或非零输入有明显退化，就不切默认版本。即使本项成功，完整分数数组和候选收集仍在；下一项按实测决定是否改善收集/累加或压缩无贡献文章。若改输入整理方式，CPU/NPU应使用一致规则，整理与传输成本单独报告。BM25S独立对照和完整查询性能仍未完成。

建议commit备注：`feat: 增加Radix零分汇总独立实验及验收`
