import onnx
import onnx_graphsurgeon as gs

from .analysis import _analyze_group
from .assembly import _apply_group, _build_group

_TARGET_OPSET = 11


def _remove_constant_nodes(graph):
    constant_nodes = [node for node in graph.nodes if node.op == "Constant"]

    for node in constant_nodes:
        assert len(node.outputs) == 1
        output_tensor = node.outputs[0]

        value_attr = node.attrs["value"]
        assert isinstance(value_attr, gs.Constant)

        output_tensor.to_constant(values=value_attr.values)
        output_tensor.inputs.clear()

    if constant_nodes:
        graph.nodes = [node for node in graph.nodes if node.op != "Constant"]

    return len(constant_nodes)


def _ensure_toposorted(nodes):
    index = {id(node): node_index for node_index, node in enumerate(nodes)}
    for node_index, node in enumerate(nodes):
        for tensor in node.inputs:
            for producer in tensor.inputs:
                producer_index = index.get(id(producer))
                if producer_index is None:
                    continue
                assert producer_index < node_index, "graph nodes are not topologically sorted"


def rewrite_model(model, groups):
    model = onnx.version_converter.convert_version(model, _TARGET_OPSET)
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)

    total_constant_count = _remove_constant_nodes(graph)
    if total_constant_count:
        print(f"Found {total_constant_count} Constant nodes; removed them before splitting.")

    orig_nodes = list(graph.nodes)
    _ensure_toposorted(orig_nodes)

    for group_cfg in groups:
        group_info = _analyze_group(orig_nodes, group_cfg, graph.outputs)
        new_nodes, stitched_exit = _build_group(group_info)
        _apply_group(graph, orig_nodes, group_info, new_nodes, stitched_exit)

    _ensure_toposorted(graph.nodes)
    out_model = gs.export_onnx(graph)
    out_model = onnx.shape_inference.infer_shapes(out_model)
    onnx.checker.check_model(out_model)

    return out_model
