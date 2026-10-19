# 原版CPU参考，独立于NPU运行

`bm25s_original/`来自上游仓库https://github.com/xhluca/bm25s，固定提交`03eac5a8fb3aa4958ce37415a4cf2220d35157f7`。

其中25个文件按Git原始blob导出，保留源码、LICENSE、README和setup.py，未修改原版实现。`source.json`由本项目添加，记录固定提交、原包Git树及逐文件SHA256。没有复制工作区`.idea`、缓存、安装环境或黄区文件。

只有`standalone/run_bm25s_reference.py`显式加载这份参考；NPU入口不导入它，不需要安装BM25S。不要用setup.py安装，也不覆盖环境中其他BM25S。报告以提交和源码指纹识别版本；原版`__version__`可能读取已有pip分发元数据或显示0.0.0，不用它代替源码身份。

首次运行与计时边界见 [BM25S离线对照步骤](../standalone/BM25S离线对照步骤.md)。此参考是原版预计算CPU路线，不是自研Ascend C算子组成部分。
