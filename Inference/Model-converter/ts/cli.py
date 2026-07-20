import argparse
from pathlib import Path

import onnx

from ts.config import parse_config
from ts.rewrite import rewrite_model
from ts.verify import verify_model


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input ONNX model")
    parser.add_argument("config", help="Path to split configuration JSON")
    parser.add_argument("output", help="Path to output ONNX model")
    parser.add_argument(
        "--ts-method",
        choices=["auto", "dupnas", "tinyts", "patchts", "nots"],
        default="auto",
        help="TS method. auto infers from ONNX filename.",
    )
    return parser.parse_args()


def _infer_ts_method_from_name(path):
    name = Path(path).name.lower()
    for method in ("dupnas", "tinyts", "patchts", "nots"):
        if method in name:
            return method
    return "unknown"


def main():
    args = _parse_args()

    ts_method = args.ts_method
    if ts_method == "auto":
        ts_method = _infer_ts_method_from_name(args.input)

    print(f"[TS] inferred method: {ts_method}")

    model = onnx.load(args.input)
    groups = parse_config(args.config)
    rewritten = rewrite_model(model, groups, ts_method=ts_method)
    onnx.save(rewritten, args.output)

    ok, diffs = verify_model(model, rewritten)
    for name, diff in diffs.items():
        print(f"{name}: max_abs_diff={diff}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
