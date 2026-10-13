"""沿用IndexSDK的ATC单算子调用形式；需要真实CANN及已安装的算子包。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "fixtures/contribution/singleop.json"))
    parser.add_argument("--soc-version", required=True, help="经核实的真实SoC参数")
    parser.add_argument("--output", default=str(PACKAGE_ROOT / "results/contribution_models"))
    parser.add_argument("--atc", default="atc")
    args = parser.parse_args()
    config = Path(args.config).resolve()
    spec = json.loads(config.read_text(encoding="utf-8"))
    if len(spec) != 1 or spec[0]["op"] != "Bm25Contribution":
        parser.error("不是贡献算子描述")
    atc = shutil.which(args.atc)
    if atc is None:
        parser.error("未找到ATC，请先加载黄区CANN环境")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("模型输出目录已存在，请指定新目录，以免误用旧模型")
    output.mkdir(parents=True)
    command = [atc, f"--singleop={config}", f"--soc_version={args.soc_version}",
               f"--output={output}", "--log=info"]
    with (output / "atc.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode != 0 or not list(output.glob("*.om")):
        raise RuntimeError(f"ATC失败或没有生成OM，查看 {output / 'atc.log'}")
    print("模型已生成：", output)
    print("尚未执行NPU。下一步编译主机调用程序并核对实际输出。")


if __name__ == "__main__":
    main()
