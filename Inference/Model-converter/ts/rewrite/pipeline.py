import onnx
import onnx_graphsurgeon as gs

from .analysis import _analyze_group
from .assembly import _apply_group, _build_group

_MIN_SUPPORTED_OPSET = 11


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


def _normalize_ts_method(ts_method):
    method = (ts_method or "").lower()
    if "dupnas" in method:
        return "dupnas"
    if "tinyts" in method:
        return "tinyts"
    if "patchts" in method:
        return "patchts"
    if "nots" in method:
        return "nots"
    return method or "unknown"


def rewrite_model(model, groups, ts_method="unknown"):
    ts_method = _normalize_ts_method(ts_method)
    # Keep the old strict check for DupNAS. TinyTS/PatchTS may create groups
    # where a non-sink output also feeds an outside consumer, so those outputs
    # are reconstructed with Concat and rewired for external users.
    allow_external_non_sink_outputs = ts_method in {"tinyts", "patchts"}

    # Preserve newer ONNX opsets. The ONNX version converter does not
    # reliably support downgrading models such as opset 12/13 to opset 11
    # (for example, Constant and Relu may have no downgrade adapter).
    current_opset = next(
        (
            int(opset.version)
            for opset in model.opset_import
            if opset.domain in ("", "ai.onnx")
        ),
        None,
    )
    if current_opset is None:
        raise RuntimeError("Cannot determine the model's default ONNX opset")

    print(
        f"[TS] input ONNX opset={current_opset}, "
        f"minimum supported opset={_MIN_SUPPORTED_OPSET}"
    )

    if current_opset < _MIN_SUPPORTED_OPSET:
        print(
            f"[TS] upgrading ONNX opset "
            f"{current_opset} -> {_MIN_SUPPORTED_OPSET}"
        )
        model = onnx.version_converter.convert_version(
            model,
            _MIN_SUPPORTED_OPSET,
        )
    else:
        print(
            f"[TS] keeping ONNX opset {current_opset}; "
            "version conversion skipped"
        )

    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)

    total_constant_count = _remove_constant_nodes(graph)
    if total_constant_count:
        print(f"Found {total_constant_count} Constant nodes; removed them before splitting.")

    orig_nodes = list(graph.nodes)
    _ensure_toposorted(orig_nodes)

    print(
        f"[TS] method={ts_method}, "
        f"allow_external_non_sink_outputs={allow_external_non_sink_outputs}"
    )

    for group_cfg in groups:
        group_info = _analyze_group(
            orig_nodes,
            group_cfg,
            graph.outputs,
            allow_external_non_sink_outputs=allow_external_non_sink_outputs,
        )
        new_nodes, stitched_outputs = _build_group(group_info)
        _apply_group(graph, orig_nodes, group_info, new_nodes, stitched_outputs)

    _ensure_toposorted(graph.nodes)
    out_model = gs.export_onnx(graph)

    # GraphSurgeon should preserve the imported model opset. Verify it here so
    # the rewritten model is never silently written as an older opset.
    output_opset = next(
        (
            int(opset.version)
            for opset in out_model.opset_import
            if opset.domain in ("", "ai.onnx")
        ),
        None,
    )
    if output_opset is None:
        raise RuntimeError("Rewritten model has no default ONNX opset")

    if output_opset != current_opset and current_opset >= _MIN_SUPPORTED_OPSET:
        print(
            f"[TS] restoring rewritten model opset "
            f"{output_opset} -> {current_opset}"
        )
        for opset in out_model.opset_import:
            if opset.domain in ("", "ai.onnx"):
                opset.version = current_opset
                break

    out_model = onnx.shape_inference.infer_shapes(out_model)
    onnx.checker.check_model(out_model)

    final_opset = next(
        int(opset.version)
        for opset in out_model.opset_import
        if opset.domain in ("", "ai.onnx")
    )
    print(f"[TS] rewritten ONNX opset={final_opset}")

    return out_model
