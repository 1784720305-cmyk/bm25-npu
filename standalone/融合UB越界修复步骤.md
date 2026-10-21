# 融合算子UB越界：修复与重新验收

## 发生了什么

2026-10-21设备5的实际日志确认：主机程序编译、两个API库及4个符号检查成功；第一次A查询、debug=1在同步时失败，返回507035。设备报告Bm25FusedScoreTopK访问UB越界，code341。没有通过融合设备正确性，也没有融合性能成绩。保留原工程、失败输出目录和日志。

可以把UB理解为每个计算核旁边的临时工作台。程序实际使用50,816字节，却只在启动时申请32,768字节；第34,048字节处已有控制数据，因此即使每块只有3篇文章，也可能访问申请范围之外。这是本轮查到的确定配置错误，与日志相符；是否还有其他问题，仍须重新上机验证。

[官方SetLocalMemorySize说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/basicdataapi/atlasopapi_07_00250.html)明确该参数表示动态UB，而不是SIMT缓存预留。当前9.2工程已支持这个接口，本次保留接口并修正数值；官方也说明新接口名称为SetDynUBufSize。

## 代码改了什么

输入仍为常驻原始TF索引和查询词编号，输出仍为局部Top-K及debug分数；FP32公式、累加顺序、选择算法、同分规则及实际aclnn签名不变。

- Host的`Tiling`把动态UB声明改为65,536字节，覆盖实际50,816字节。编译期断言检查容量与对齐，设置失败立即返回错误。
- 将32 KiB最低DCache和8 KiB编译器预留单独列入保守容量检查；这是容量预算，不是通过该接口分别配置缓存。预算合计106,496字节，不能再写成“50,816＋32 KiB局部空间”。
- 工程元信息增加`ub_contract_version=2`及上述独立字段。运行脚本拒绝旧工程，避免源码更新但仍调用旧配置。
- 新增23项从实际C++常量推导的布局/容量检查，明确拒绝旧32 KiB声明；调用保护117项、计算模型311项通过。它们不证明CANN编译、真实SIMT同步或设备正确性。

设备内核只修正内存说明注释，计算代码不变；现有主机C++三文件、旧验收算子、CPU、二进制样本和离线wheel均不改。主机接口不变，已编译的`bm25_fused_chain_runner`可复用。

## 黄区分步操作

本地提交推送之后，黄区先拉取。每段成功才执行下一段；不要复制日志或提示符作为命令。

```bash
cd /test/w50066029/bm25-npu
git pull --ff-only
python3 verify_package.py
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -o pipefail
python3 standalone/create_fused_score_topk.py --compute-resource ai_core-ascend950 --msopgen-python python3 --output results/Bm25FusedScoreTopKUbFix1
```

生成成功后编译新目录，保留旧目录：

```bash
cd /test/w50066029/bm25-npu/results/Bm25FusedScoreTopKUbFix1
bash build.sh 2>&1 | tee ../fused_score_topk_ubfix1_build.log
```

编译成功后，安装新融合包，更新同一个融合vendor。无需重建其余算子或主机程序：

```bash
bash build_out/custom_opp_openEuler_x86_64.run --install-path=/test/w50066029/bm25-npu/results/custom_opp
```

安装成功后启动新的验收进程。必须显式传入新`--project`；结果和日志也用新名称：

```bash
cd /test/w50066029/bm25-npu
python3 standalone/run_fused_chain.py --input results/query_score_small_input1 --output results/fused_chain_small_s2_ubfix1 --project results/Bm25FusedScoreTopKUbFix1 --device 5 --k 3 --shards 2 2>&1 | tee results/fused_chain_small_s2_ubfix1.log
```

设备5如被占用，改为空闲卡；不使用别人在用的卡4。若结果名已经存在，换后缀，不覆盖。

## 如何验收

先通过5批A/B/A/空/B，每批debug开启和关闭两种模式：完整分数与独立公式在原容差内一致，局部/最终编号严格正确，所选FP32分数字节正确，两种模式和重复查询结果相同，保护区及常驻索引不变。成功报告应为`fused_chain_actual_debug_modes_and_all_rankings_passed`。

只有上述实际设备检查通过，才补S64、64词及最大分区等集中边界，随后接正式Search并比较性能。此次不重跑旧版性能采集或CPU基线。若仍有设备错误，停在新失败日志定位，不能视为已修复、跳过检查或使用CPU替代结果。

提交备注：`fix: 修正950融合算子动态UB声明与缓存预留`。
