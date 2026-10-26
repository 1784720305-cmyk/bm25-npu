# BM25 A5 自研计算原型

**最新实机状态：Graph＋目录版六轮CPU目标未达到；当前短采集已完成，下一轮只优化NPU算子，CPU实现冻结。** 用户提供的正式20次采集显示，A/B融合算子约82.7565/83.0425µs、合并约5.4195/5.315µs、链内间隔0.022µs，尚不能拆分融合内部评分/选择耗时。新增[寄存器内阈值选择](standalone/融合算子寄存器选择步骤.md)，消除四轮选择中间CDF写回和逐轮标量二分/同步；468组本地精确模型及来源、CPU命令冻结、失败即停检查通过，尚未CANN编译或设备执行。黄区只构建独立候选算子，复用现有主机、两个CPU二进制和已有利用率采集，正确性通过后按原六轮完整Search门槛判断，不提前宣称超过CPU。

上一轮普通基准：NPU A/B六轮批次中位数的中位数约101.5/101.4µs；每轮取两种CPU更快者后，A/B约83.4µs，较快轮次仍约66µs；NPU分别只赢2/6、3/6轮。CPU部分p95有毫秒级波动，不能借慢轮次宣称达标。原始汇总见[设备结果记录](standalone/graph_directory_device_evidence_20261026.json)。[当前组合短采集](standalone/Graph目录版短采集步骤.md)已完成，不重复采集；旧比例不外推到新组合。

**2026-10-26 最新状态：SIMD 融合候选已由用户在 A5 通过编译、正确性和完整 Search 对照，之前的 UB/命名空间失败已经修复。** 256K/Q8/K10 同期六轮 A：CPU32/32 70.71125µs、NPU 140.691µs；B：CPU 69.339µs、NPU 140.46825µs，NPU 均 0/6 轮获胜。此前相对旧 NPU 约 24% 的改善成立，完整 Search 超过 CPU 的目标仍未达到。

已实现并完成设备执行的[常驻执行计划、异步传输与分区索引](standalone/BM25_A5端到端优化方案_20261026.md)：固定锁页 arena 和张量，捕获 H2D/两个算子/D2H，每次更换查询内容并重放、最终等待一次；用全词静态分区目录消除重复边界查找，CPU 使用同等目录。原[闭环步骤](standalone/Graph常驻执行计划闭环步骤.md)已经完成，不重复执行；55µs 仍是未达到的目标。旧源码审计见[证据记录](standalone/a5_search_optimization_evidence_20261026.json)。以下各段为历史阶段记录，状态以最上方最新结果为准。

**当前主线：修复融合算子的动态UB声明，再验收设备正确性。** 黄区新主机编译及4个API符号归属检查已通过；设备5首次融合执行在同步时失败507035，设备报告UB越界code341，尚无融合正确性或性能结论。查到启动只声明32,768字节，却使用50,816字节；现源码改为64 KiB动态UB，并单独检查缓存/编译器预留。23项布局、117项调用保护及311项计算模型检查通过，不代替实机。按[修复步骤](standalone/融合UB越界修复步骤.md)新建工程、重建并安装融合算子，复用已编译主机程序；保留失败记录。旧三段约0.333ms、CPU32线程约0.074ms，性能目标未达成。

**历史过程：下述266/271us属于更换评分及Top-K之前，当前组合瓶颈须由新采集判断。** 256K/Q8的105条设备任务与主机样本交叉通过；正式A评分265.986us、局部Top-K270.701us、合并5.3425us，设备链跨度中位数541.67us，采集下完整655.813us，逐次计算链外112.245us。这些是采集诊断，不能相加阶段中位数或与普通基准相减。普通六轮原查询A：64K/Q1 CPU0.0146245/NPU0.0763145ms，256K/Q8 CPU0.07011525/NPU0.579449ms，A/B均NPU0/6轮更快。本次新增连续文章分区/前进游标的独立评分源码，复用已验收CPU的文章分区思路；原算子、Search、计时/采集程序和样本不改。实验每查询最多64词；用户已在950设备2安装并实际执行新vendor，小样本及4097文章的A/B/A/空/B会话均通过独立公式、索引一次上传/回读不变和重复字节检查。4097文章A有65条查询、最长51词，最大绝对误差3.8146973e-05，按原相对/绝对容差通过，并非每条65词。三组64词/连续范围跨块/尾部补验后续也已在设备2通过，共15次实际评分，65词仅验证主机拒绝。新评分后续已通过完整Search门禁及六轮新旧对照：256K/Q8原查询A旧0.5826435ms、新0.39418125ms，下降32.346%；B旧0.587098ms、新0.39268625ms，下降33.114%，两组新版均快6/6轮。保留新版评分，不能据此宣称超过CPU/BM25S。下一项只改局部Top-K：用户核对已有结果得每8192文章分块约730正分，A最大800/B最大825，两组256/256块均能放入1024候选。独立正分压缩Top-K后续已编译、安装，实际接口及37组集中设备验收均通过：设备4共185次调用，包含1023/1024/1025容量、正负零、同分真实编号、稠密完整路径和交替覆写。局部Top-K完整Search对照随后已在设备4通过：A原0.38595125ms/压缩版0.3321105ms，下降13.95%；B原0.3821755ms/压缩版0.3317205ms，下降13.20%，两组均压缩版快6/6轮。保留新组合作为候选，尚未证明超过CPU。随后同期六轮已完成：A CPU0.0742165ms/NPU0.332506ms、B CPU0.07401325ms/NPU0.3325485ms，NPU均快0/6轮，约慢4.48/4.49倍，结果核对通过、性能目标未达成。本轮补固定原版BM25S Numba的同边界原生/严格适配Search入口；本地74项假模型边界检查及实际Numba0.68串并行集成通过，目标Numba0.60尚待黄区执行。原评分/Top-K/合并、Search和验收源码不改，最终性能目标尚未达成。见[连续文章分区评分实验步骤](standalone/连续文章分区评分实验步骤.md)。


**最终search要求已固定：CPU已有查询词编号后开始计时，到CPU收到合规Top-K后停止；见[计时与验收合同](standalone/search计时与验收合同.md)。** NPU包含每次查询整理、上传、索引检索、计算、同步和结果下载；CPU采用相同输入/输出边界，也允许索引提前常驻内存。约0.37ms的resident只作计算实验；2.24092ms的旧含传输实验仍未计CPU展开，不能改称完整search。必须让整库索引常驻并在NPU按新查询取数，消除逐查询展开/搬运，而不是从计时中删除。当前新索引格式及评分算子源码已准备，评分会话已在设备4小样本实跑并通过五批独立公式及索引复用核对，同边界CPU入口已通过小样例实机与NPU交叉；原版BM25S包装器在开始计时前的未知词/空查询兼容处理，最终也要移入search计时。

**2026-10-20主线调整：按MT要求改为整库索引长期保存＋每批只传查询词编号。** 具体输入、步骤、输出和验收见[整库常驻索引路线](standalone/整库常驻索引路线.md)。本次完成倒排索引/独立查询包的主机格式、全TF/公式核对和182项本地检查；保留现有已验收算子；独立评分算子和同进程调用器已编译，小样本五批设备评分正确性通过；三段流程现已通过小样本实机排名和同进程换查询验收；四项K边界和较大批量三段补验也已通过；原始词编号NPU Search后续已编译运行并核对通过，当前编译独立CPU索引Search；正式稳态比较仍未完成。

用户日志已确认Numba0.60.0/llvmlite0.43.0黄区私有安装及CPU编译/并行小检查通过；这是CPU依赖就绪，原版BM25S Numba检索尚未运行，不再重复安装。以下参考对照和旧阶段记录保留，以本主线为准。

**成熟CPU参考进展：黄区原版BM25S的NumPy参考已跑通64K/Q1和256K/Q8；原版Numba目标依赖已在黄区私有安装、CPU编译/并行小检查通过，实际BM25S检索和性能尚未验收。** 256K/Q8完整分数及严格适配排名通过，原版5条查询不满足同分编号规则，单独记录。原版1/8查询工作线程中位数47.026/9.34476ms，严格适配27.5896/6.20007ms；不把这轮NumPy参考称为最佳BM25S，也不替代更快的C++CPU对手来计算大倍数NPU加速。

本次按 [Numba离线步骤](standalone/BM25SNumba离线步骤.md) 新增可选原版编译评分、串行/并行检索、编译签名验证及首次编译/预热分离记录。25个原BM25S文件及原NPU/C++/设备验收字节不变。NumPy仍为默认，原版同分差异与外部严格适配分别测；选择Numba后依赖/编译失败即停，不自动回退。目标候选Numba0.60.0/llvmlite0.43.0按官方兼容表选择，NumPy2.0.2保留，需实际黄区验收。安装器只允许项目results下新目录、离线/无依赖解析/强制SHA256，随后CPU JIT小检查；不改CANN或系统依赖。

本机Python3.10/NumPy2.2.6/Numba0.68.0原版串并行及严格适配、公式通过，249项NumPy回归和38项离线管理失败保护通过；后38项使用假wheel/模拟安装，不是实机安装证据。外网Windows已实际下载Numba0.60.0及llvmlite0.43.0两只CPython3.9/Linux x86_64 wheel，大小/PyPI SHA256/归档/版本检查通过，两个原始二进制与lock已纳入170文件清单。`optional_offline_dependencies.numba.state=downloaded_verified_not_installed`；随Git提交新增offline_dependencies及清单，用户黄区check、私有install及目标CPU JIT/并行小检查现已通过；BM25S查询正确性和性能尚未验证。清单字段描述发布包本身，不作为黄区安装状态证据。Windows的python命中启动器且用户终端无输出，直接调用已验证的pythoncore-3.10-64/python.exe后下载完成；不据此断言启动器内部故障原因。完整操作见上述步骤。

NPU候选收集版的已测优势边界保留：256K/Q8/K10驻留计算相对当前8线程C++CPU赢6/6轮，256K/Q1只赢1/6轮；含传输256K/Q8首项中位数2.24092ms仍未证明完整查询优势。按MT要求转向整库索引常驻＋每批查询词编号，成熟CPU参考继续补齐；本次先固定并检查新输入格式。原始黄区文件留黄区，仅分享允许的终端汇总。

本次提交备注建议：`fix: 修正950融合算子动态UB声明与缓存预留`。

下文旧阶段描述和数字保留作历史记录，当前状态与下一步以上述说明及新实验步骤为准。

官方参考的最新实机状态：主机编译和库加载成功，513×40/K=32通过；65×1025/K=10在完整排名检查失败，首个差异涉及正负零同分成员。保留失败结果，不放宽规则，暂停其后续性能实验。

**最近自研实机结果（2026-10-16）：新Warp独立13组、四段小样本、65536文章三轮CPU交叉及本次四段采集已由用户在黄区通过。设备2本次正式10轮：贡献2.143us、累加5.484us、局部Top-K40.643us、最终合并4.266us；逐轮设备总和中位数52.6us、链跨度54.46us。当前仍无稳定超过合理CPU的证据，局部Top-K仍为主要耗时。下一步按上述IndexSDK借鉴路线设计独立自研选择算子，先正确性再比较。**

[官方Top-K对照步骤](standalone/官方TopK对照步骤.md) 保留作历史参考，当前暂停执行；它不被自研主流程引用。黄区文件留黄区，只分享允许的运行摘要。以下较早实验数字和步骤保留为历史记录，以以上最新决定和IndexSDK借鉴路线为准。

**当前性能推进顺序已重排：先诊断新版和CPU实际资源，再建立正常查询的规模/批量曲线，按瓶颈选择融合算分或跨块Top-K，随后接常驻索引与完整查询。详见[950性能优化路线](standalone/950性能优化路线.md)。当前约1.5倍是相对指定4线程CPU的驻留计算初测，不作为最终性能交付。**

实际主机已确认是双路AMD EPYC 9575F，共128物理核/256逻辑CPU，当前进程亲和性允许256逻辑CPU；此前仅指定4个CPU工作线程，不能称为整机CPU性能。下一步先按实际msprof帮助采集新版设备和运行库时间，再做合理CPU线程配置扫描。

msprof第一轮已在设备1实际采集到三算子记录，op_summary有51条设备任务。`standalone/analyze_pipeline_profile.py` 已由用户在黄区只读核对CSV及运行元信息，排除首次/预热后汇总正式10轮，避免把前几行或初始化/退出API耗时当作稳态，见[性能路线](standalone/950性能优化路线.md)。

正式10轮分析已由用户在黄区完成：贡献8.123us、累加28.768us、Top-K43.37us；逐轮三段设备耗时之和中位数80.319us，链内间隔合计中位数0.044us。当前优先研究Top-K。已补正常短查询数据模式和 `standalone/benchmark_case.py` 自动执行合理CPU线程扫描及真实NPU对照，先测65536文章的单查询/32查询；新工作负载的实机结果尚待获取。见性能路线内命令及计时范围。

新版并行Top-K已在A5编译、安装并通过小样本与4096文章设备运行。设备4的17查询整批resident中位数为0.090416ms；4线程C++ CPU按相同输入、指定预热5/30轮复测为0.131157ms，两次中位数比约1.45。全部170个检索编号及顺序严格一致，数值在原容差内。结论仅限当前输入和4线程CPU的驻留计算初测；合理CPU线程扫描、同卡新旧复测和完整查询仍待推进。详见[并行Top-K实机记录与步骤](standalone/并行TopK优化步骤.md)。

**最新路线：自研Ascend C算子 + 实际自动生成的aclnn接口直接调用，不经过ATC或OM，不依赖IndexSDK/Faiss。先读 [standalone/ACLNN路线.md](standalone/ACLNN路线.md)。原SDK接入、ATC与ACL句柄步骤保留为历史参考，当前不执行。**

这是可独立上传Git、再传入黄区的源码快照。包含自研Ascend C贡献、累加、Top-K源码、三个已通过小样本实机核对的调用器、数据工具与样本；历史SDK接入工具仅保留作参考。

**状态：新版并行Top-K已通过6查询小样本及4096文章/17查询的设备正确性、重复与CPU直接交叉。当前计时为已准备输入的驻留计算实验；新版边界、统一条件新旧/CPU性能复测、更多规模、合理CPU线程扫描、单条/批量、含传输和完整BM25S比较仍待推进。**

后续复测旧版设备1为4.49012ms、新版设备4为0.087912ms，二者设备结果检查通过；设备号不同，暂不报告同卡新旧优化倍数。用户说明设备4正在被他人使用，后续避开设备4，复用设备1旧版记录，补新版设备1及该次输出的直接CPU交叉。

## 先理解这一步

输入是每条记录的词频TF、文章长度项、IDF；设备计算 `IDF * (TF * 2.2) / (TF + 长度项)`，固定k1=1.2，输出每条记录的贡献。每个线程写不同输出，不涉及文章总分的并发累加。

这些展开输入便于第一次核对公式。后续完整查询接回稀疏倒排数组；如果由CPU展开输入，正式完整查询计时必须包含展开工作。

## 目录

| 位置 | 内容 |
|---|---|
| npu_prototype/op_host | 贡献/累加/Top-K注册、输入输出和任务分配 |
| npu_prototype/op_kernel | 自研A5 SIMT贡献、累加与Top-K |
| npu_prototype/topk_parallel | 并行Top-K的注册/任务分配、候选缓冲及组内归并源码；小样本及4096文章A5运行通过 |
| npu_prototype/topk_merge_host、standalone/validate_topk_merge.py | 新合并算子独立调用，设备2七组边界/重复验收通过 |
| npu_prototype/host | 三个算子的aclnn直接调用器及CMake，均已通过小样本实机核对 |
| npu_prototype/sharded_host、standalone/run_sharded_pipeline.py | 新四段实验：候选在NPU产生并合并，设备2小样本和4096文章/17查询/S=8设备运行、重复及CPU交叉通过；该配置尚未展示速度收益 |
| npu_prototype/pipeline_host | 诊断、普通resident重复及分段模式已实机通过；新版库来源和额外工作区保护已在A5运行核对 |
| cpu_reference | 优化C++/OpenMP贡献、累加、候选堆Top-K；黄区GNU10.3.1/OpenMP4.5编译和4线程正确性通过 |
| standalone/scale_fixture.py、run_cpu.py | 可复现规模输入准备与实际C++ CPU调用；复用四输出核对 |
| standalone/run_topk.py | 从已准备元信息读取尺寸/指纹，启动实际C++ Top-K调用器 |
| standalone/validate_topk.py | 自动执行剩余三项Top-K边界/重复实机补验，失败停止并保存报告 |
| standalone/pipeline_fixture.py、run_pipeline.py | 串联输入准备、四个输出核对及实际调用入口 |
| standalone/run_pipeline_benchmark.py、compare_compute_outputs.py | 新增实际预热/重复NPU调用与所有查询CPU/NPU直接交叉检查 |
| npu_prototype/*.py | 生成/校验/合入SDK材料、生成ATC模型；可选样本重建 |
| fixtures/contribution | 三组输入、期望值、22条记录说明和singleop.json |
| fixtures/compute_contract_small.* | 原始小合同样本，供可选重建使用 |
| check_npu_environment.py | 环境查询，无第三方依赖 |
| check_result.py | 样本及设备输出检查，无第三方依赖 |

不包含IndexSDK、BM25S、虚拟环境、已编译模型和大数据。主运行路线需要匹配的CANN及自研算子安装包，不依赖IndexSDK。核心统计/分词不要求放到NPU，Python可准备数据。

## 本机/黄区先运行

以下命令从本仓库根目录执行，Python工具路径不依赖原来的Windows目录。

```bash
python3 verify_package.py
python3 check_result.py
python3 check_npu_environment.py --output results/npu_environment.json
```

第一条检查快照文件指纹，第二条核对已打包样本，第三条只查询工具/设备并写报告。没有NPU环境时报告会显示未就绪。

普通运行和检查不需要NumPy。只有重新生成样本才需要Python3.10及以上与requirements.txt中的NumPy；黄区如不能联网，应使用组内已有环境/获准的离线依赖，已打包样本可直接使用。

## 旧方案：接入组内SDK（当前不执行）

修改以下路径为黄区真实的、属于你的SDK工作副本：

```bash
SDK_ROOT=/实际路径/IndexSDK
python3 npu_prototype/prepare_sdk_overlay.py --sdk-root "$SDK_ROOT"
python3 npu_prototype/apply_sdk_overlay.py --sdk-root "$SDK_ROOT"
python3 npu_prototype/apply_sdk_overlay.py --sdk-root "$SDK_ROOT" --apply
```

准备工具只生成4个覆盖文件和指纹。第二条Python命令只核对；第三条才备份原CMake并合入。已有同名文件、文件指纹或SDK源码列表不匹配会停止，不能覆盖别人改动。重复执行需先检查已有接入情况。

加载组内CANN环境后，在SDK自己的工作副本构建：

```bash
cd "$SDK_ROOT/feature_retrieval/src/ascendfaiss/ops"
bash build.sh Ascend950
```

该组内入口会重建工作副本的build_out。生成算子包后按组内实际安装/加载流程处理，确认Bm25Contribution注册和实现映射正确。本文不假定固定安装包名称或权限。

## 旧ATC/OM方案（当前不执行）：生成模型和编译调用器

回到本仓库根目录。SoC和TOOLKIT_HOME应按实际环境确认；下方SoC是当前SDK的映射示例：

```bash
python3 npu_prototype/generate_model.py --soc-version Ascend950PR_957c
cmake -S npu_prototype/host -B results/host_build -DTOOLKIT_HOME=/实际CANN工具包目录
cmake --build results/host_build -j
```

模型入口使用已打包的singleop.json，输出results/contribution_models并保存ATC日志。要求新目录，避免误用旧模型。主机调用器使用C++17，建议g++9以上；旧编译器需要按环境调整。

此节描述旧ACL句柄路线。当前host源码已改为aclnn，接口和命令均见standalone/ACLNN路线.md，不再接受模型目录参数。

## 旧ACL句柄方案（当前不执行）：执行并检查

```bash
results/host_build/bm25_contribution_runner results/contribution_models fixtures/contribution results/actual_run1.bin 0
python3 check_result.py --actual results/actual_run1.bin
```

最后的0是实际设备编号。输出需新文件，保存执行日志。通过标准是设备实际执行成功，并且22条贡献在容差范围内一致。

程序打印单次提交/同步与上传/执行/下载耗时，只用于诊断，不是预热后的性能报告，不包含完整BM25累加与Top-K。

## 向MT解释关键函数

- ContributionSimt：每个线程读取记录，用自研公式算贡献。
- Tiling：依据输入长度和设备核数分配任务。
- Read / Require：读输入、检查长度与ACL错误。
- Runtime / Resources：管理设备、内存和句柄，退出前等待并释放。
- 当前调用器管理的是aclTensor与workspace，通过两阶段aclnn执行，不创建ACL单算子句柄或读取OM。
- runner的main：上传→执行→同步→下载；不读取CPU期望值。
- check_result：独立检查公式/文件，再核对设备输出。

下一阶段：在相近负载、统一预热/轮数下复测同设备新旧版和CPU，补新版边界、合理CPU线程扫描，再扩大规模、批量并完成含传输及原版BM25S完整查询两类性能比较。

## 上传Git

本目录可以作为新的Git仓库根目录。已有.gitignore会排除输出、缓存和环境，保留22条小样本。

```bash
git init -b main
git add .
git commit -m "Prepare A5 BM25 contribution prototype"
git remote add origin <你的仓库地址>
git push -u origin main
```

仓库地址需自行填写；本次未连接远端、未提交或推送。不要把外层整个bm25工作目录上传。

## 参考与声明

工程接入参考组内IndexSDK的CAGRA/IVFPQ A5实现；数值样本对照过原版BM25S。两个项目源码未打包进来。脚本生成的覆盖材料可能包含SDK原CMake内容，按组内仓库和许可证要求处理。

package_manifest.json记录此源码快照的原始文件字节SHA256，方便确认传输是否完整。文本统一为LF，.gitattributes要求Git检出也使用LF；.bin/.npz保持二进制原始字节，校验器不忽略换行差异。

修改代码后先保持文本为LF，再执行 `python3 rebuild_manifest.py` 和 `python3 verify_package.py`。重建工具拒绝含CR换行的文本，不会改写二进制样本。新增发布文件需明确加入清单文件列表，缓存和运行输出不在其中。
