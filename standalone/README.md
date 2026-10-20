# 独立CANN工程路线（当前主线）

2026-10-20当前主线：整库常驻索引＋每次查询词编号，按[最终search合同](search计时与验收合同.md)计时。评分算子及调用器已编译，在设备4连续五批评分及索引复用核对通过；当前按[分块边界步骤](常驻评分边界验收步骤.md)补更大尺寸正确性，不必重编重装。Top-K接入及search性能尚未完成。下文旧算子步骤仅保留复现，不作为当前进度。

最新状态：贡献/累加小样本已通过，设备2的Top-K 6×6/K=3、513×40/K=32及三项补验全部通过，见[Top-K记录](A5TopK验证记录.md)。当前已补[同进程串联调用程序](同进程串联步骤.md)，等待黄区编译、接口加载和设备正确性验证；已有算子安装包继续使用。

运行路线为 [自研算子+aclnn直接调用](ACLNN路线.md)，不再生成ATC/OM。本文的工程生成步骤供首次创建工程参考，已有工程不必重新生成。

不依赖IndexSDK源码、库、安装包或Faiss。IndexSDK仅作为SIMT写法与正确性组织方式的参考。贡献公式和主机ACL调用器是我们自己的代码。

采用设备环境自带的官方msopgen生成配套框架，避免把其他版本的构建脚本当成当前CANN要求；`create_project.py`再把自己的Host/SIMT实现放进去，配置ascend950和独立vendor。使用官方工程工具不等于调用第三方BM25算法。

## 第一件事：确认黄区生成器

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
command -v msopgen
msopgen gen --help
```

如果命令不在PATH，查询位置，不需要IndexSDK：

```bash
find /usr/local/Ascend/cann-9.2.0 -maxdepth 7 -type f -name msopgen
```

把工具路径、帮助或错误发回后确认其支持的A5 `ai_core-...` 目标。不把框架的ascend950、ATC的Ascend950PR_957c、生成器的Compute Resource当成相同参数。

## 生成工程

目标确认后，在仓库根目录执行；尖括号为占位符，不要原样运行：

```bash
python3 standalone/create_project.py --compute-resource <已确认的ai_core目标>
```

生成目录默认是results/Bm25Contribution。已有目录时停止，防止覆盖。工具生成完整CMake框架，脚本检查预期结构，替换三个自研源文件、更新CMakePresets，保留模板备份。结构不匹配就报错，需要查看实际生成工程再适配。

## 编译与执行

```bash
cd results/Bm25Contribution
bash build.sh
```

构建应产出独立custom_opp安装包。位置、安装/加载选项以该版本生成工程与包帮助为准；不要安装未经核实的旧SDK包。实际编译错误要逐项修复，特别是SIMT API和tiling数据生成规则。

算子包正确加载后，读取实际生成的aclnn头文件，按真实签名编写/编译直接调用器，执行并用check_result.py检查。不运行原generate_model.py，不使用OM。

## 代码与验收

输入：三组等长FP32（TF/长度项/IDF）。处理：自研BM25贡献公式。输出：每条贡献。验收：设备日志证明A5真实执行，22条数值与CPU一致。

`populate`只处理新生成的模板工程和自研源文件；`main`负责启动生成器。它们不计算BM25、不安装软件、不运行设备，不把工具返回成功当设备正确性。

本机只检查脚本、数据与模板替换；三个算子的小样本设备执行由黄区日志确认通过。完整查询仍需要扩展验证和设备串联；尚无正式性能结论。

参考：

- [CANN9.2工程编译部署](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/asc-devkit/docs/zh/guide/programming_guide/advanced_programming/aclnn_operator_development/compilation_and_deployment/basic_process.md)
- [msopgen工程创建说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/tbeaicpudevg/atlasopdev_10_0024.html)
