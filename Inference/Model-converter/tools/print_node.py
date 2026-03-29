import argparse
import onnx

ap = argparse.ArgumentParser()
ap.add_argument("model", help="path to .onnx")
ap.add_argument("--no-constants", action="store_true")
args = ap.parse_args()

m = onnx.load(args.model)
nodes = list(m.graph.node)

if args.no_constants:
    nodes = [node for node in nodes if node.op_type != "Constant"]

for idx, node in enumerate(nodes):
    name = node.name if node.name else "<unnamed>"
    print(f"{idx:4d}\t{node.op_type}\t{name}")
