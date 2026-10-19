# BM25 A5 自研计算原型

**最新进展（2026-10-19）：零分汇总版已由用户在黄区完成编译、独立安装、实际接口/依赖核对、23组共115次独立设备验收，以及小样本和256K文章/8查询的四段运行、首末/重复六数组及CPU直接交叉。** 全部80个最终编号和顺序一致，原FP32输出规则保留。大样本本次预热2次/测量3次诊断中位数0.417896ms，与原版历史时间接近，不能据此判定优化收益；尚未证明稳定加速。

当前只新增 [新旧Radix同期对比入口与步骤](standalone/Radix新旧同期对比步骤.md)：同一输入、设备、已编译四段调用器及其他三个算子，六轮轮换先后，每轮预热5次/测量30次，独立公式与完整排名核对，新旧/跨轮六数组字节一致。21项本地顺序/来源/计时/失败控制检查通过，是模拟工具检查，不是新旧NPU性能结果。已验收的算子、独立验收器、调用器、CPU和样本均不改，无需重编或重跑23组。比对入口实机运行与性能结论待取得。

此前更快8线程CPU与原Radix在256K/8查询基本持平（六个中位数再取中位数：CPU 0.4169955ms、NPU 0.413235ms）。原版设备分段贡献6.413us、累加86.572us、Top-K301.586us、合并5.145us，整链跨度中位数400.589us，只作采集诊断。CPU最佳配置、BM25S、含传输和完整查询性能仍未完成。

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
