# A5 常驻执行计划与静态分区目录：一次执行入口

状态：源码和本地 Python 检查已完成；尚未在目标 CANN 编译、运行或证明超过 CPU。本机没有 A5；用户确认仍由其在黄区终端执行。

本次固定路线为：已验收 SIMD 算子 → 固定锁页输入/输出与张量 → 整链异步传输和 Graph 重放 → 静态分区目录。保留旧可用版本，独立目录 vendor，不通过不断改线程数、桶数或测试规模推进。

## 已实现

- `common/fixed_query_packet.h`：Search 内直接整理新的原始查询；固定容量、完整覆盖尾部、保留词序和重复词；非法输入在修改缓冲前拒绝。
- `common/shard_directory.h`：全词 int64 分区边界，与请求无关。CPU 和 NPU 在各自 session 冷初始化时构建一次。
- `npu_prototype/graph_search_host/`：锁页及设备连续 arena、固定张量、异步与 Graph 路径；整条 H2D/评分选择/合并/D2H 重放后等待一次；返回对象复制和结果检查仍在 Search 内。
- `standalone/shard_directory_sources.py`：从冻结源生成独立目录候选，只改首个输入张量的形状合同和两次边界查找；函数 ABI、FP32 公式、词序、SIMD 选择和合并规则保持原样。
- `cpu_reference/directory_search/`：给原 CPU 算法同等目录预处理，编译参数保持 `-O3 -march=native -ffp-contract=off -fno-fast-math`。
- `standalone/graph_search.py`：编译/实际 ABI 检查、受影响设备验收、完整 Search 对照及最终 CPU 门槛。

索引、输入、输出、候选缓冲有设备保护区；捕获异常时先结束捕获，再销毁模型及引用资源。无法确认在途工作已结束时保留整组资源直到进程退出，不在模型仍可能引用内存时释放一部分。

Graph 缓存任务结构。每次新输入的写入、上传、计算、下载和输出复制全部真实执行并计时。正式 Search 不下载完整分数，完整分数只在未计时的检查点核对。

## 本地发布

仓库根是 `F:\Ascend\bm25\bm25-a5`，不是外层目录。源码未自动提交或推送。确认本次修改后可执行：

```powershell
cd F:\Ascend\bm25\bm25-a5
git add README.md package_manifest.json common cpu_reference/directory_search npu_prototype/graph_search_host standalone/graph_search.py standalone/shard_directory_sources.py standalone/test_graph_search.py standalone/a5_search_optimization_evidence_20261026.json "standalone/BM25_A5端到端优化方案_20261026.md" "standalone/Graph常驻执行计划闭环步骤.md"
git commit -m "实现A5常驻Graph检索与CPU同等静态分区目录闭环"
git push origin main
```

## 黄区只运行这个入口

默认沿用此前实际路径、设备 5 和已完成的 `results/fused_simd_close2`。输出必须是新目录，已有目录不覆盖。

```bash
cd /test/w50066029/bm25-npu &&
git pull --ff-only &&
source /usr/local/Ascend/cann-9.2.0/set_env.sh &&
python3 verify_package.py &&
set -o pipefail &&
python3 -u -B standalone/graph_search.py \
  --output results/graph_directory_close1 --device 5 \
  2>&1 | tee results/graph_directory_close1.log
```

脚本固定完成以下阶段，不要求逐个手动改配置或替换源码：

1. 核对原 256K/Q8/K10 语料、已验收 SIMD 工程/vendor 和来源指纹。
2. 用实际 CANN 头文件和库验证捕获接口可编译链接，再编译新主机；运行 C++ 打包/CPU 目录逐字节交叉检查。
3. 克隆已有原始工程，生成、编译并安装独立 `bm25_fused_score_topk_directory`；检查实际生成 ABI 和设备目标文件。原 SIMD vendor 不覆盖。
4. 小样本对异步、Graph、Graph＋目录三条路径各运行八次换查询检查：A/B/A/空/未知词/64词/空/B。每次有全分数、严格局部和最终排名、原始分数字节、缓冲保护区和索引回读。
5. 原已验收 NPU 路径、固定异步路径、Graph 路径各做一次完整 Search 对照，辨别开销来源。这三次不作为超过 CPU 的验收。
6. 六轮同期交替运行原 CPU32/32、目录 CPU32/32、Graph＋目录 NPU；每轮预热 5 次、正式 30 次 A/B。采用每轮更快的 CPU 作对手，同时要求 p95 汇总改善。所有正式进程保留完整输入、样本、检查点及二进制指纹。

默认不改变当前最好 CPU32/32。若此前另有更快合理线程配置，应使用 `--threads` 指定至少 32 的相应配置并如实报告；不能降低线程数使 NPU 获胜。Q1、Q32 和真实语料需要独立范围验收，当前通过也只说明原 Q8 主负载达标。

## 如何判断结果

- 退出码 `0`：`actual_graph_directory_full_search_faster_than_cpu`。A/B 都六轮全胜，p95 汇总也优于两个 CPU 对手中更快者，并通过所有正确性检查。
- 退出码 `2`：`actual_graph_directory_correct_but_cpu_target_not_met`。正确性通过，但性能没有达到目标；不作为达标交付。
- 退出码 `1`：编译、捕获、正确性或来源检查失败，立即停止，保留首个错误及对应阶段日志，不回退 CPU 伪造 NPU 结果。

55µs 是工程目标，不是写死的通过值。CPU 更快时必须面对更快的 CPU。完整报告在 `results/graph_directory_close1/report.json`；各阶段日志也在同目录。原始黄区文件仍留黄区，只分享允许的终端汇总或首错。

本地已运行八项 Python 检查，覆盖源变更范围、空分区/非整除分区、独立小样本回读和 FP32 参考、六轮/最快 CPU/p95 门槛、来源清单及计时结构；这些不是 C++ 编译、设备验收或性能证据。C++、实际 CANN 和设备测试由上述入口统一完成。
