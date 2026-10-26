# 已验收 Graph＋目录版：一次短采集，不重建

2026-10-26 用户日志确认：337文件版已在设备5完成编译、便携检查、三条路径各八次设备验收，以及六轮同期CPU/NPU对照。性能目标未达成，原失败报告保留。NPU A/B整批时间约101.5/101.4µs，较快CPU轮次约66µs；不能选择CPU慢轮次来宣称成功。

本轮只新增 Python 采集/分析工具及记录。现有 NPU/CPU 源码、二进制、算子安装、索引和计时边界都不改；不再编译安装或重跑六轮 CPU。完整终端摘要及计算见 `graph_directory_device_evidence_20261026.json`。

## 为什么只采集当前组合

一次归因对照中，旧路径约143–145µs，固定异步约119µs，Graph约118µs。Graph/目录正式六轮约100–102µs。不能拿这些不同进程的差值当作某段精确耗时；但它们已说明“继续改Host提交就必然能补齐剩余差距”缺乏依据。

这次要确认当前设备上的评分选择融合算子、最终合并、两者间隔，以及同次完整Search在计算链以外的时间。融合算子内部评分和选择尚不能由这次两个kernel的采集直接拆分。链外包含传输、Host、运行时和采集影响，不能当作纯拷贝耗时。

## 发布

本地仓库 `F:\Ascend\bm25\bm25-a5`：

```powershell
git add README.md package_manifest.json standalone/profile_graph_directory.py standalone/test_profile_graph_directory.py standalone/graph_directory_device_evidence_20261026.json "standalone/Graph目录版短采集步骤.md" "standalone/Graph常驻执行计划闭环步骤.md" "standalone/BM25_A5端到端优化方案_20261026.md"
git commit -m "增加已验收Graph目录版短采集并记录CPU未达标结果"
git push origin main
```

## 黄区执行

```bash
cd /test/w50066029/bm25-npu &&
git pull --ff-only &&
source /usr/local/Ascend/cann-9.2.0/set_env.sh &&
python3 verify_package.py &&
set -o pipefail &&
python3 -u -B standalone/profile_graph_directory.py \
  --completed results/graph_directory_close1 \
  --output results/graph_directory_profile1 \
  2>&1 | tee results/graph_directory_profile1.log
```

入口读取已经完成的闭环报告，逐字节核对旧主机、实际安装库/设备目标和索引，复用该报告中的实际命令，仅把输出改到新目录、正式轮数从30缩到10；仍预热5轮。设备编号沿用已完成报告的5，不猜设备或替换程序。读取实际 msprof 帮助后启用此前已确认的 runtime-api、task-time=l1、PipeUtilization 配置。

当前主机会提交35次公开Search加4条额外debug链，共78条计算任务；只统计20次正式A/B。检查点与已经验收的完整输出逐字节核对，不重新构造CPU性能对手。

Graph可能复用Task ID，也可能将不同节点放在不同执行Stream。分析器按设备、Stream、Task ID和开始时间匹配每次执行，再核对时长、任务类型、输入/输出形状和调用顺序。包括冷启动在内，输入词张量形状都必须保持容量24，首个索引张量必须是V×33目录，不能沿用旧非Graph分析器的“冷启动words=1”规则。

缺行、重复实例不能唯一匹配或CSV模式变化时，保存原始数据并明确停止，不猜测轮次或套用旧版比例。结果写入 `results/graph_directory_profile1/device_analysis.json`，终端打印A/B两算子、链跨度、间隔及链外耗时。性能判断仍保留本次普通六轮的未达标结论。

本地八项合成测试已通过，覆盖重复Task ID、不同Stream、冷启动/预热/debug排除、缺失或歧义实例、形状和时间错配；不是实际msprof执行证据。参考[官方采集数据说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0042.html)及[官方性能数据字段说明](https://github.com/mindstudio-docs/master/blob/master/msprof/docs/zh/user_guide/profile_data_file_references.md)。
