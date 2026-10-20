# 从小样例推进到正常规模Search

## 已通过什么，这次补什么

用户黄区日志确认，CPU常驻索引Search在GNU10.3.1/OpenMP4.5编译成功，180次独立FP32公式/严格排名/交替调用检查，以及16项错误输入/失败停止检查通过。小语料A/B/A/空/B五次CPU Search与已有NPU Search结果直接核对全部通过。CPU整批诊断为0.007261、0.006700、0.006200、0.005048、0.006239ms；不是稳态成绩，不拿它与不同时间采集的NPU五次日志直接计算加速比。

现在CPU和NPU两个公共入口都已在小语料跑通。下一步保留这两个可执行程序与所有算子，**只新增正常规模数据准备和联合验收脚本**，不需要重新编译、安装NPU或CPU。本轮尚未执行正常规模CPU/NPU Search，也没有新的性能结论。

## 输入、处理、输出和验收

输入采用已有正常合成语料目录，例如`results/normal_64k_q1`中的原始`corpus.json`、`metadata.json`和独立公式`expected_scores.bin`。不读取旧的逐查询TF、IDF、长度项、分组offset数组。文章和原查询不重新随机生成，也不改写原目录。

准备程序做三件事：

1. 从原始文章建立覆盖全部词的原始TF倒排索引。索引提前加载，不预计算BM25贡献。
2. A完整保留原短查询；B将每个词编号加1后对词表大小取余，查询条数和长度不变，用于验证同一索引连续换查询。空批保持相同查询条数，仅用于正确性，不列为正常性能负载。
3. 独立从原文章统计TF/DF，用原BM25公式生成期望，并核对索引格式的FP32结果。A还要与旧语料独立公式文件逐字节一致。复制这三份来源文件，使新输入自带来源核对，不依赖旧目录继续存在。

联合验收程序实际先执行C++ CPU，再执行NPU。两端分别在同一进程加载一次索引，然后按A/B/A/空/B调用五次。接入方式与小样例一致；**CPU返回或检查失败后停止，不启动后续NPU，不用CPU输出代替设备输出**。输出分开存为`cpu/`和`device/`。

验收要求：完整总分与独立公式在原容差内；CPU与同输入有序FP32公式逐字节一致；NPU局部候选和最终排名严格正确；CPU/NPU每条查询编号和顺序严格一致、分数在原容差内；重复查询输出字节一致；索引只加载/上传一次并核对不变；真实主机程序的编译源码指纹与当前源码一致。不能只检查文件尺寸或前几条排名。

两端Search仍遵守[计时合同](search计时与验收合同.md)：从原始分组词编号开始，NPU包括整理/上传/索引检索/计算/同步/Top-K下载，CPU包括自身必需的索引检索/计算/Top-K，到CPU取得合规结果停止。本工具记录五次单次诊断，不增加预热/正式测量轮次，也不计算性能倍数。调试污染/完整分数导出/独立检查在Search外；各次之间有调试操作，所以不能据这五次推断稳态。

## 第一步：65536篇文章、单查询

本次只增加Python工具，原来的CPU和NPU程序已编译通过，不用重编。提交推送后在黄区逐条执行，任何命令失败就停；已有输出目录请换新名字，不删除历史报告。

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -o pipefail

python3 standalone/search_scale_fixture.py prepare --input results/normal_64k_q1 --output results/search_64k_q1_input1

python3 standalone/run_search_scale.py --input results/search_64k_q1_input1 --output results/search_64k_q1_cpu_npu_check1 --threads 8 --cpu-document-shards 8 --shards 32 --device 2 2>&1 | tee results/search_64k_q1_cpu_npu_check1.log
```

预期数据尺寸：N=65536，A/B各Q=1，K=10，每条3词。已有本机同语料实际准备通过，索引16,766,264字节、每条逻辑查询28字节、返回Top-K80字节。这只是明确应用数组的字节数，不包含框架内部通信，也不由字节减少推算延迟改善。

默认容量取本样本最大查询数/词数，双方一致：Q1/3词；不再默认预分配65条查询。空请求有一个不被读取的物理INT32占位，应用查询H2D为20字节，而普通3词查询为28字节。容量启动的合成空请求及冷启动单列。

工具默认沿用已经验收的`results/search_host_build/bm25_search_runner`、`results/cpu_index_search_build/bm25_cpu_index_search_runner`，三个原安装vendor和`results/radix_collect_topk_validation1`。若实际目录不同，使用`--npu-runner`、`--cpu-runner`、`--vendors`或`--topk-acceptance`，不跳过来源检查。选择当前空闲卡，不使用被占用的设备4；示例继续设备2。

## 第二步：第一项通过后，再做262144篇文章、8查询

```bash
python3 standalone/search_scale_fixture.py prepare --input results/normal_256k_q8_batch1 --output results/search_256k_q8_input1

python3 standalone/run_search_scale.py --input results/search_256k_q8_input1 --output results/search_256k_q8_cpu_npu_check1 --threads 8 --cpu-document-shards 8 --shards 32 --device 2 2>&1 | tee results/search_256k_q8_cpu_npu_check1.log
```

此项A/B各8条3词正常查询，逻辑上传168字节，K10返回640字节；索引字节数以实际准备日志为准。完整原始语料、索引还原、全分数和排名检查会比小语料慢，程序会打印当前阶段。查询数×文章数上限为16,777,216，仍须另考虑Python语料/索引/独立参考及CANN工作区内存，不把该上限当峰值内存证明。

成功报告`report.json`的状态应为`normal_scale_cpu_npu_raw_id_search_and_all_checks_passed`，五项`checks`均包含CPU、NPU和直接交叉结果。CPU索引读取一次，NPU索引上传一次，两个进程分别启动而不是共享索引内存。原始词编号文件共同使用，没有在CPU前预先做NPU专用打包。

这里的8线程和8个CPU文章分区是正确性配置，不代表服务器最佳CPU性能。后续正式比较需要扫描合理线程/文章分区、记录可用资源与亲和性，并在相近负载下做预热和多轮交替运行。Q1报告延迟，Q8报告整批时间和吞吐，不能把Q8时间除以8当单条延迟。原版BM25S的同边界适配与Numba实机检索也要继续补齐；其预计算贡献路线与当前查询时公式分别说明。

## 代码怎么向MT解释

- `search_scale_fixture.definition`：核对旧语料来源及正常短查询声明，确定A/B/空三批词编号。未知/重复词等规则已在小样例/边界验收，正常样本不再注入压力查询。
- `independent_scores`：从原文章统计TF/DF，再算独立双精度公式，最后转FP32，作为核对文件；它不是CPU性能对手，也不写进NPU实际输出目录。
- `validate_inputs`：从复制的原语料重建索引身份、重算期望，拒绝只改哈希却与原语料不符的输入。
- `run_search_scale.run`：调用两个已实机验收的C++程序，复用已有会话记录、候选/排名和CPU/NPU直接核对函数；不改计时范围，不新增算子，不修改旧验收入口。

本机56项检查包含真实主机数据准备/核对，以及假安装和模拟CPU/NPU子进程的成功/失败路径，不是正常规模设备执行证据。原CPU/NPU工具的67项和128项模拟回归也通过。黄区原文件留黄区，只分享允许的终端汇总或首个错误。

本次建议commit备注：`feat: 增加正常规模常驻索引Search联合验收`
