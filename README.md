# BM25 A5 自研计算原型

三个自研算子均已通过小样本实机核对：贡献22条、累加36组（设备0/1输出一致）、Top-K在设备2通过6×6/K=3及513×40/K=32，后者16,416项完全一致。见[Top-K验证记录](standalone/A5TopK验证记录.md)。现在按[Top-K步骤](standalone/TopK步骤.md)运行K=1/全部文章/重复运行三项补验，再串联完整设备查询；尚无性能达标结论。

**最新路线：自研Ascend C算子 + 实际自动生成的aclnn接口直接调用，不经过ATC或OM，不依赖IndexSDK/Faiss。先读 [standalone/ACLNN路线.md](standalone/ACLNN路线.md)。原SDK接入、ATC与ACL句柄步骤保留为历史参考，当前不执行。**

这是可独立上传Git、再传入黄区的源码快照。包含自研Ascend C贡献、累加、Top-K源码、三个已通过小样本实机核对的调用器、数据工具与样本；历史SDK接入工具仅保留作参考。

**状态：依据黄区日志，贡献最大绝对误差0，累加最大绝对误差1.1920929e-07且两次跨设备输出一致，Top-K小样本排名和分数字节完全一致。大规模、完整查询和正式性能验收未完成；设备1初始化停顿与公共vendor运行库问题仍需独立定位。**

## 先理解这一步

输入是每条记录的词频TF、文章长度项、IDF；设备计算 `IDF * (TF * 2.2) / (TF + 长度项)`，固定k1=1.2，输出每条记录的贡献。每个线程写不同输出，不涉及文章总分的并发累加。

这些展开输入便于第一次核对公式。后续完整查询接回稀疏倒排数组；如果由CPU展开输入，正式完整查询计时必须包含展开工作。

## 目录

| 位置 | 内容 |
|---|---|
| npu_prototype/op_host | 贡献/累加/Top-K注册、输入输出和任务分配 |
| npu_prototype/op_kernel | 自研A5 SIMT贡献、累加与Top-K |
| npu_prototype/host | 三个算子的aclnn直接调用器及CMake，均已通过小样本实机核对 |
| standalone/run_topk.py | 从已准备元信息读取尺寸/指纹，启动实际C++ Top-K调用器 |
| standalone/validate_topk.py | 自动执行剩余三项Top-K边界/重复实机补验，失败停止并保存报告 |
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

下一阶段：扩大三个算子的正确性测试、串联设备计算，最后测相同流程CPU/NPU及完整NPU/BM25S两类比较。

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
