# 使用 Git 同步到 A5

远程仓库：`https://github.com/1784720305-cmyk/bm25-npu.git`，分支 `main`。需要 A5 能访问 GitHub，并具有该仓库的读取权限；不可联网时继续使用已提供的源码 ZIP。

## 首次使用

保留旧 `/test/w50066029/bm25-npu`，其中的数据、vendor 和历史结果继续使用。首次克隆到另一个新目录：

```bash
cd /test/w50066029
git clone --branch main --single-branch \
  https://github.com/1784720305-cmyk/bm25-npu.git bm25-npu-indexsdk
cd bm25-npu-indexsdk
git rev-parse --short HEAD
python3 verify_package.py
```

然后按 [A5 验收与日志回传](A5_ACCEPTANCE.md) 填写 A5 上实际的 IndexSDK 源码、主机库、clang-tidy 路径，先运行 host 阶段。项目根目录改为此处的 `bm25-npu-indexsdk`；vendor 和输入数据参数仍指向旧项目目录。

如果旧目录原本来自 ZIP，没有 `.git`，不能直接在其中 pull。也不要在旧目录执行强制覆盖、git clean 或 reset。

## 后续更新

```bash
cd /test/w50066029/bm25-npu-indexsdk
git status --short
```

确认没有本地源码修改后：

```bash
git pull --ff-only origin main
git rev-parse --short HEAD
python3 verify_package.py
```

`--ff-only` 遇到分叉会停止，不会自动合并。若 status 显示源码修改或 pull 提示冲突，保留现场和输出，再处理本地修改；不要直接 reset 或 clean。

`results/`、`build/`、虚拟环境和 IDE 缓存已忽略，运行数据不应提交到仓库。pull 更新源码后，C++ 改动仍需重新编译；验收使用新的 RUN，例如 `results/indexsdk_a5_acceptance_02`，不能用旧二进制和旧结果冒充新版本通过。已有历史目录不删除。

回传验收日志时附上 `git rev-parse --short HEAD`，便于对应源码版本。

2026-11-04 已完成的验收对应代码 `9a99bbc`，成功结果位于 `results/indexsdk_a5_acceptance_02/device_03`。后续文档和证据归档提交保留该版本的生产源码、构建与验收脚本指纹；已有成功测量保留即可。复现同一验收版本可使用该提交；新的源码变更仍按上述流程创建新 RUN 验收。详见 [验收记录](A5_ACCEPTANCE.md#5-2026-11-04-a5-验收记录)。
