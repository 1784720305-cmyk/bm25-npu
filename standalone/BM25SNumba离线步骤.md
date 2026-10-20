# BM25S原版Numba CPU对照：准备与状态

当前NumPy原版在黄区256K/Q8/K10完整分数和严格适配排名通过。原版5条查询存在同分编号差异，未篡改库；原版1/8查询工作线程批次中位数47.026/9.34476ms；外部严格适配1/8线程27.5896/6.20007ms。保留两条路径，不把严格适配时间记成原版API时间，不从它们相减推断修正成本。这是一轮CPU参考测量，不是最佳BM25S，也不直接与历史NPU算加速比。原NumPy评分使用`np.add.at`，不是逐文章的自写Python循环。

目前合理优化的C++CPU仍是更快的主要对手：256K/Q8此前约0.40..0.46ms；NPU含传输首测2.24092ms。不能用较慢NumPy参考替代更快CPU对手并宣布目标完成。下一步补原版提供的Numba路径，之后再设计完整NPU查询的数据流，不新增Top-K内核。

## 输入、处理、输出及验收

输入仍为现有64K/Q1或256K/Q8目录，公式、词编号、精度和同分规则不改。原BM25S25个文件及固定提交不变；仅扩展外部调用脚本的`--backend numba`。建索引仍用同一NumPy CSC构建路线，查询使用原版编译评分和原版Numba检索/选择。外部严格适配仍单独报告，包含编译评分及NumPy严格选择，不能称为完整Numba原版API。

原版Numba检索提供串行和`prange`并行实现，线程并行对象仍是多条查询；Q=1不等于8个核拆一条查询。记录请求线程数和最多同时工作的查询数。当前不以此冒充最优CPU线程配置。

首次评分/检索可能编译。脚本分别记录`score_validation_prepare_ms_including_possible_jit`、`first_native_call_ms_including_possible_jit`和各配置`initial_call_before_warmup_ms`。这些首次阶段和预热/核对均排除在稳态计时样本外。评分及原版串行/并行检索须生成`nopython_signatures`；`NUMBA_DISABLE_JIT`即使设为0也拒绝，防止绕过原版的环境分支。选择Numba后缺依赖或编译失败就停止，不退回NumPy或解释执行。

公式容差保持`1e-6+2e-6*abs(expected)`。先核对所有文章总分、原生Top-K分数/原FP32字节及同分差异，再核对严格适配的全部编号。原版Numba也不保证我们的同分规则，不放宽规则或修改原库来消除差异。保存完整样本、中位数/p95、实际后端及来源、编译签名检查状态。接口、分数或排名失败即停止。

## 版本依据和当前实际状态

[Numba官方0.60兼容表](https://numba.readthedocs.io/en/0.60.0/user/installing.html#version-support-information)列出Python3.9..3.12、NumPy2.0系列和llvmlite0.43系列，因此选择`numba==0.60.0`及`llvmlite==0.43.0`，保留黄区已有NumPy2.0.2。[官方发布说明](https://numba.readthedocs.io/en/0.60.0/release/0.60.0-notes.html)同时指出NumPy2.0存在执行语义差异，版本表不能代替本项目的逐项数值验收。

这两个CPython3.9/Linux x86_64/manylinux2014 wheel需要从外网导入，不含NumPy，不安装LLVM系统工具。二进制必须从官方PyPI下载，按其SHA256、大小、平台标签、METADATA和压缩包CRC核对，并原样保留自带许可证。下载脚本不把Windows/其他Python wheel换进来。

**两个目标wheel现已下载并通过校验，已连同lock纳入170文件发布清单；尚未在黄区安装或运行Numba。** 实际文件大小：Numba 3,695,086字节，llvmlite 43,871,777字节。PyPI SHA256、大小、平台、版本和内部CRC检查通过。此前工具联网失败及Windows启动器无输出不再作为当前“依赖未取得”的状态；用户已直接调用真正的Python解释器完成下载，未改系统NumPy或CANN。

本地Python3.10/NumPy2.2.6/Numba0.68.0上原版编译评分、原版串行/并行检索、严格适配和独立公式检查已通过；这是接口集成检查，不是目标0.60/Python3.9兼容性或性能结论。249项原NumPy回归和38项离线归档/平台/私有安装/失败保护检查通过，后38项使用临时假wheel/模拟进程，包括下载进度及连接/读取超时，不是实际下载或安装验收。

## 第一步：外网Windows取得依赖

不要在黄区执行联网下载。请在外网Windows普通PowerShell终端、本独立Git仓库内执行：

```powershell
cd F:\Ascend\bm25\bm25-a5
& 'C:\Users\kojo-jotaro\AppData\Local\Python\pythoncore-3.10-64\python.exe' -u -B -X utf8 standalone/prepare_bm25s_numba.py download
```

它只向固定版本的官方PyPI地址发起下载，写入`offline_dependencies/numba_py39_linux/`；不安装任何东西、不修改NumPy/CANN。成功后会生成两只wheel与`lock.json`，自动更新并重建原始字节清单、运行校验，不执行Git提交。

启动时立即显示保存目录，然后显示正在查询的PyPI地址；文件下载期间约每秒显示一次已收到的字节数（收到新数据时更新）。连接/读取超时设置为30秒，这不是整次下载或系统DNS解析的总时限。失败信息带阶段、地址和原始错误；不会静默切换镜像、禁用证书检查或改版本。旧脚本在取得版本信息前没有提示，容易被误认为没启动；若旧命令仍在等待，先在那个终端按一次`Ctrl+C`，更新脚本后再执行上述命令，避免多个下载进程同时写文件。

下载失败停止，已有但摘要不符的文件不覆盖，未完成文件保留`.part`便于诊断，不加入清单。成功时应有“两个包已按PyPI SHA256核对并加入清单”的汇总；`optional_offline_dependencies.numba.state`从`awaiting_download`变成`downloaded_verified_not_installed`。将新增文件及更新的清单一起按现有Git方式传入黄区，不仅提交脚本。

当前这一步已实际完成，不必重新下载。将两只wheel、lock和更新的源码清单随Git传入黄区；下面先做黄区核对和私有安装，通过后再运行Numba检索。

## 后续第二步：黄区核对和私有安装

拉取包含实际两包的提交后：

```bash
cd /test/w50066029/bm25-npu
python3 verify_package.py
python3 standalone/prepare_bm25s_numba.py check
```

通过后安装到项目新目录：

```bash
python3 standalone/prepare_bm25s_numba.py install --output results/bm25s_numba_py39
```

安装器先确认Linux x86_64、CPython3.9、NumPy2.0.2及glibc>=2.17；通过`pip --no-index --no-deps --require-hashes --target`只安装已核对两包。不改系统包、不导出全局PYTHONPATH、不影响NPU调用。新进程随后核对实际加载位置及精确版本，做FP32 CPU编译/并行小检查，记录安装文件SHA256。此时仍不是BM25S验收。

目录存在就停止，不覆盖历史记录；安装或CPU编译失败保留`installation.json`和探测错误，不记成功，选新目录重试前先定位原因。

## 后续第三步：先小场景验证原版Numba

私有依赖成功后：

```bash
set -o pipefail
python3 standalone/run_bm25s_reference.py run --backend numba --dependency-dir results/bm25s_numba_py39 --input results/normal_64k_q1 --output results/bm25s_numba_64k_q1_ref1 --threads 1 8 --warmup 5 --repeats 30 2>&1 | tee results/bm25s_numba_64k_q1_ref1.log
```

脚本只在本CPU参考进程里加载该私有目录；运行前后核对文件及lock，原NPU命令不加载它。成功状态应为`original_bm25s_numba_and_separate_strict_adapter_checks_completed`，并且原版编译签名及完整数值/排名检查全部通过。原生同分差异仍独立列出。

64K确认后才扩大到256K/Q8：

```bash
python3 standalone/run_bm25s_reference.py run --backend numba --dependency-dir results/bm25s_numba_py39 --input results/normal_256k_q8_batch1 --output results/bm25s_numba_256k_q8_ref1 --threads 1 8 --warmup 5 --repeats 30 2>&1 | tee results/bm25s_numba_256k_q8_ref1.log
```

之后选更快CPU配置，保留原NumPy、C++及全部结果。BM25S预计算路线与NPU查询时算公式仍分开解释，原生同分规则未满足时不当作严格规则检索对手。正式硬件结论仍须同条件重复进程测量，以及NPU输入准备/传输与完整查询边界；不从单次报告宣布“超过BM25S最优”。

提交备注建议：`feat: 增加BM25S原版Numba入口及私有离线依赖准备`
