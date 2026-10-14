# 当前主运行路线：自研Ascend C + 自动生成aclnn

最新进展：贡献、累加、Top-K的aclnn调用及设备1同进程串联均已编译和运行成功，小样本全部结果核对通过。最新证据见 [A5串联验证记录.md](A5串联验证记录.md)，当前下一步见 [扩大验证步骤.md](扩大验证步骤.md)。下面保留贡献阶段的接口说明，正式性能尚未验收。

不依赖Faiss或IndexSDK，IndexSDK仅作A5实现参考。原版BM25S继续单独用于数值核对与CPU性能对照。

## 实际状态（根据黄区日志）

- 已在Ascend950PR、CANN 9.2.0环境生成独立工程，贡献算子已编译、打包、安装。
- 安装目录：`/test/w50066029/bm25-npu/results/custom_opp/vendors/bm25`。
- 产物含`op_api/include/aclnn_bm25_contribution.h`和`op_api/lib/libcust_opapi.so`。
- ATC连`--help`也无法启动：libgert缺少`aclrtHostGetDevicePointerAddrRange`。检查过的runtime/runtime_common/ascendcl未导出该符号。
- 尚未执行贡献算子，尚未验证设备结果，更没有性能结论。

ATC/OM路线退出主线。原generate_model.py与ACL句柄调用器只是旧诊断材料，当前不要执行。新调用器必须以实际生成头文件为依据，不能根据名称猜函数签名。

## 接口与依赖证据（已收到黄区输出）

实际生成头文件确认：

```cpp
aclnnStatus aclnnBm25ContributionGetWorkspaceSize(
    const aclTensor *tf, const aclTensor *docNorm, const aclTensor *idf,
    const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnBm25Contribution(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

头文件依赖`aclnn/acl_meta.h`。收到的`ldd -r libcust_opapi.so`输出未显示not found或undefined symbol，依赖含同一CANN目录的libnnopbase、libascendcl、libacl_rt和runtime等。未直接显示libgert，但运行中仍可能动态加载其他库，不能据此保证绕开环境问题。

在黄区项目根目录：

```bash
cat results/custom_opp/vendors/bm25/op_api/include/aclnn_bm25_contribution.h
ldd -r results/custom_opp/vendors/bm25/op_api/lib/libcust_opapi.so > results/aclnn_dependencies.log 2>&1
cat results/aclnn_dependencies.log
```

cat读取实际函数声明、头文件依赖；ldd -r检查动态库加载及未解析的函数，不执行BM25算子。把输出带回，核对包含的类型、参数顺序、workspace/executor/stream契约与运行时库来源，再修改调用器和CMake。

本机没有完整CANN开发环境，接口依据是用户贴出的黄区真实头文件。调用器直接include已安装头文件，不打包替代头文件；编译时static_assert校验上述两个函数指针签名。

## 随后的实现顺序

输入仍是22条记录的TF、doc_norm、IDF；贡献算子源码与二进制样本保持不变。

1. 基于实际头文件创建输入/输出张量，链接生成的op_api库及已核对的CANN依赖。
2. 输入上传后调用执行准备接口，取得执行对象与临时内存大小。
3. 按实际大小分配workspace（大小为0时不无条件分配），调用执行接口。
4. stream同步后下载结果，检查有限值并保存到新输出文件。
5. 按接口的实际所有权约定释放张量、临时内存、设备缓冲、stream/context等；失败时停止，并等待已提交工作结束再释放相关资源。
6. 用check_result.py与expected.bin核对，保存环境、构建和执行日志。调用器不读取expected.bin，也不在CPU生成替代结果。

正常调用使用非复用executor，第二阶段由框架释放；不调用用于可复用executor的销毁接口。若准备成功后因workspace分配或执行失败中断，本诊断程序直接退出，不复用残留executor；用户资源仍按安全顺序清理。未经过第二阶段的框架内部对象在进程结束时回收，这不是可持续服务的失败恢复实现。

## 最新调用器的函数逻辑

| 函数 | 逻辑 |
|---|---|
| Read | 读取三组输入文件，检查FP32长度 |
| Resources::Allocate | 分配Device内存，0字节返回nullptr；记录待释放资源 |
| Resources::Describe | 用aclCreateTensor创建一维连续FP32张量，stride=1，offset=0 |
| Require | 检查API返回值，失败抛出错误并停止 |
| Runtime/Resources::Close | 等待stream完成，释放张量、内存及运行时资源；释放失败不保存结果 |
| main | 校验输入→上传→GetWorkspaceSize→分配workspace→执行→同步→下载→释放→保存 |

输出先写NaN检测漏写，调用器不读取expected.bin。输出文件必须新建；错误日志通过tee保存，设备结果单独用check_result.py检查。单次诊断耗时含打印，不是性能基准。

## 黄区下一步：编译新调用器

更新Git到本次代码版本后，从`/test/w50066029/bm25-npu`执行；贡献算子源码/二进制样本未改，不需要为本次主机改动重新生成和安装算子包：

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
source /test/w50066029/bm25-npu/results/custom_opp/vendors/bm25/bin/set_env.bash
python3 verify_package.py
set -o pipefail
cmake -S npu_prototype/host -B results/aclnn_host_build \
  -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 \
  -DBM25_OPP_ROOT=/test/w50066029/bm25-npu/results/custom_opp/vendors/bm25 \
  2>&1 | tee results/aclnn_configure.log
cmake --build results/aclnn_host_build -j 2>&1 | tee results/aclnn_build.log
```

上一条失败时停在该阶段，不继续下一条。使用新的build目录，以免旧ACL句柄工程的CMake缓存造成混用。CMake只查指定CANN/vendor目录，链接libcust_opapi、libnnopbase、libascendcl；不使用Faiss/IndexSDK/ATC。

编译成功后先检查可执行程序实际依赖：

```bash
ldd -r results/aclnn_host_build/bm25_contribution_runner > results/aclnn_runner_dependencies.log 2>&1
cat results/aclnn_runner_dependencies.log
```

没有缺库/缺符号后，选组内允许使用的设备编号执行（下面0为示例），保存日志：

```bash
results/aclnn_host_build/bm25_contribution_runner \
  fixtures/contribution results/actual_aclnn_run1.bin 0 \
  2>&1 | tee results/aclnn_run1.log
python3 check_result.py --actual results/actual_aclnn_run1.bin \
  2>&1 | tee results/aclnn_check1.log
```

执行失败时不运行结果检查，保留完整日志定位。若缺失符号在main之前出现，阶段日志不会打印；若在GetWorkspaceSize或execute阶段出现，则阶段日志帮助定位。ATC的环境问题继续保留，不修改系统库来制造假接口。

## 运行库问题继续定位

直接aclnn免去ATC模型生成，但op_api执行也可能依赖相同库。ldd -r若发现缺符号或not found，先定位库来源、版本及导出接口。没有报错也只能说明这一步通过，必须实际执行后判断是否可用。不要修改系统库来伪造缺失接口，不使用CPU结果掩盖执行失败。

## 验收分层

代码完成：真实签名匹配、路径无需OM、错误和资源释放逻辑齐全。

本次已完成此调用器源码与编译配置，本机仅做文件/源码检查，尚未实际CANN编译运行。黄区此前成功编译的是贡献算子包，不是本次新调用器。

编译成功：A5环境的头文件和库真正能编译链接调用器。

设备运行正确：aclnn实际执行、同步成功，22条贡献对齐，日志和结果文件配套。

性能达标：贡献/累加/自研Top-K全部完成后，按统一计时边界和CPU对手正式测试。当前不进入此阶段。

贡献正确后才继续文章分数累加和自研Top-K。每个阶段先解释输入、步骤、输出和验收，再实现和报告真实状态。

基础API参考：[aclCreateTensor与acl_meta](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)、[executor非复用生命周期说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/aolapi/operatorlist_00021.html)。编译与设备行为仍以当前CANN实际头文件、库和运行日志为准。
