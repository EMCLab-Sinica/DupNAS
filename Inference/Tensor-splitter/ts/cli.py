import argparse

import onnx

from ts.config import parse_config
from ts.rewrite import rewrite_model
from ts.verify import verify_model


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input ONNX model")
    parser.add_argument("config", help="Path to split configuration JSON")
    parser.add_argument("output", help="Path to output ONNX model")
    return parser.parse_args()


def main():
    args = _parse_args()

    model = onnx.load(args.input)
    groups = parse_config(args.config)
    rewritten = rewrite_model(model, groups)
    onnx.save(rewritten, args.output)

    ok, diffs = verify_model(model, rewritten)
    for name, diff in diffs.items():
        print(f"{name}: max_abs_diff={diff}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
