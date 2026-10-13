"""准备第一步算子输入，或检查实机输出；不执行或模拟 NPU。"""
import argparse
import json
from pathlib import Path

import numpy as np


def prepare(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with np.load(source, allow_pickle=False) as data:
        if not np.isclose(float(data["k1"]), 1.2, rtol=0, atol=1e-7):
            raise ValueError("首次算子原型固定k1=1.2，输入参数不匹配")
        expected_full = np.zeros_like(data["expected_scores"])
        for query in range(len(data["query_offsets"]) - 1):
            a, b = data["query_offsets"][query:query + 2]
            for term in data["query_terms"][a:b]:
                start, end = data["term_offsets"][term:term + 2]
                for pos in range(start, end):
                    doc = int(data["doc_ids"][pos])
                    rows.append((query, doc, float(data["tf"][pos]),
                                 float(data["doc_norm"][doc]), float(data["idf"][term])))
        if not rows:
            raise ValueError("这一步的贡献算子需至少一条记录；全空查询在调用层跳过贡献阶段")
        tf = np.array([r[2] for r in rows], dtype="<f4")
        norm = np.array([r[3] for r in rows], dtype="<f4")
        idf = np.array([r[4] for r in rows], dtype="<f4")
        expected = (idf * (tf * np.float32(2.2))) / (tf + norm)
        independent = np.array([r[4] * (r[2] * 2.2) / (r[2] + r[3]) for r in rows])
        np.testing.assert_allclose(expected, independent, rtol=2e-6, atol=1e-6)
        for row, value in zip(rows, expected):
            expected_full[row[0], row[1]] += value
        np.testing.assert_allclose(expected_full, data["expected_scores"], rtol=2e-6, atol=1e-6)
    for name, array in (("tf", tf), ("doc_norm", norm), ("idf", idf), ("expected", expected)):
        array.astype("<f4").tofile(directory / f"{name}.bin")
    model = [{"op": "Bm25Contribution",
              "input_desc": [{"format": "ND", "shape": [len(rows)], "type": "float32"} for _ in range(3)],
              "output_desc": [{"format": "ND", "shape": [len(rows)], "type": "float32"}], "attr": []}]
    (directory / "singleop.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    (directory / "records.json").write_text(json.dumps({
        "record_count": len(rows), "k1": 1.2, "dtype": "little_endian_float32",
        "rows": [{"query": r[0], "doc": r[1], "tf": r[2], "doc_norm": r[3], "idf": r[4]}
                 for r in rows],
        "status": "CPU fixture checked; NPU not executed",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"准备完成：{len(rows)}条记录；与独立公式、文章总分对齐。目录：{directory}")
    print("这是调试时展开的输入，不是完整查询的最终数据路径或性能结果。")


def check(directory, actual_path):
    expected = np.fromfile(directory / "expected.bin", dtype="<f4")
    actual = np.fromfile(actual_path, dtype="<f4")
    if actual_path.stat().st_size != expected.nbytes:
        raise ValueError("输出文件字节数错误，应为记录数乘4")
    if not np.all(np.isfinite(actual)):
        raise ValueError("输出含 NaN 或无穷大")
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
    print(f"输出数值检查通过，最大绝对误差：{float(np.max(np.abs(actual - expected))):.8g}")
    print("此检查只核对文件数值；是否来自NPU须结合实机运行日志确认。")


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", default=str(PACKAGE_ROOT / "fixtures/compute_contract_small.npz"))
    p.add_argument("--directory", default=str(PACKAGE_ROOT / "results/regenerated_contribution"))
    c = sub.add_parser("check")
    c.add_argument("--directory", default=str(PACKAGE_ROOT / "results/regenerated_contribution"))
    c.add_argument("--actual", required=True, help="实际运行导出的float32二进制结果")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(Path(args.source), Path(args.directory))
    else:
        check(Path(args.directory), Path(args.actual))


if __name__ == "__main__":
    main()
