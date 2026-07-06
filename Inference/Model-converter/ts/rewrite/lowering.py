import numpy as np
import onnx_graphsurgeon as gs

_MAX_CONCAT_INPUTS = 10


def _infer_spatial_axis(shape):
    assert shape is not None and len(shape) >= 4
    return len(shape) - 2


def _emit_bounded_concat(inputs, axis, output_name, output_shape, node_name, output_dtype):
    concat_nodes = []
    stage_inputs = list(inputs)
    stage_index = 0

    while len(stage_inputs) > _MAX_CONCAT_INPUTS:
        next_stage_inputs = []
        for group_index, start in enumerate(range(0, len(stage_inputs), _MAX_CONCAT_INPUTS)):
            group_inputs = stage_inputs[start : start + _MAX_CONCAT_INPUTS]
            axis_sizes = [tensor.shape[axis] for tensor in group_inputs]
            group_axis_size = None if any(size is None for size in axis_sizes) else sum(axis_sizes)
            group_shape = list(output_shape)
            group_shape[axis] = group_axis_size
            group_name = f"{node_name}_s{stage_index}_g{group_index}"
            group_out = gs.Variable(f"{group_name}_out", dtype=output_dtype, shape=group_shape)
            concat_nodes.append(
                gs.Node(
                    name=group_name,
                    op="Concat",
                    inputs=group_inputs,
                    outputs=[group_out],
                    attrs={"axis": axis},
                )
            )
            next_stage_inputs.append(group_out)
        stage_inputs = next_stage_inputs
        stage_index += 1

    out = gs.Variable(output_name, dtype=output_dtype, shape=output_shape)
    concat_nodes.append(
        gs.Node(
            name=node_name,
            op="Concat",
            inputs=stage_inputs,
            outputs=[out],
            attrs={"axis": axis},
        )
    )
    return out, concat_nodes


def _emit_entry_tiles(entry_tensor, entry_ranges, name_scope):
    axis = _infer_spatial_axis(entry_tensor.shape)
    scope_prefix = f"{name_scope}__"

    tiles = []
    slice_nodes = []
    for tile_id, ((start_h, end_h), (start_w, end_w)) in enumerate(entry_ranges):
        out_shape = list(entry_tensor.shape)
        out_shape[axis] = end_h - start_h
        out_shape[axis + 1] = end_w - start_w

        base_name = f"{scope_prefix}{entry_tensor.name}_split{tile_id}"
        out = gs.Variable(
            f"{base_name}_out",
            dtype=entry_tensor.dtype,
            shape=out_shape,
        )
        slice_node = gs.Node(
            name=base_name,
            op="Slice",
            inputs=[
                entry_tensor,
                gs.Constant(f"{base_name}_starts", np.array([start_h, start_w], dtype=np.int64)),
                gs.Constant(f"{base_name}_ends", np.array([end_h, end_w], dtype=np.int64)),
                gs.Constant(f"{base_name}_axes", np.array([axis, axis + 1], dtype=np.int64)),
                gs.Constant(f"{base_name}_steps", np.array([1, 1], dtype=np.int64)),
            ],
            outputs=[out],
        )

        tiles.append(out)
        slice_nodes.append(slice_node)

    return tiles, slice_nodes


def _emit_tile_crop(tile, produced_range, required_range, split_id, name_prefix, name_scope):
    axis = _infer_spatial_axis(tile.shape)
    scope_prefix = f"{name_scope}__"

    (prod_y0, _), (prod_x0, _) = produced_range
    (req_y0, req_y1), (req_x0, req_x1) = required_range

    rel_start_h = req_y0 - prod_y0
    rel_end_h = req_y1 - prod_y0
    rel_start_w = req_x0 - prod_x0
    rel_end_w = req_x1 - prod_x0

    out_shape = list(tile.shape)
    out_shape[axis] = req_y1 - req_y0
    out_shape[axis + 1] = req_x1 - req_x0

    base_name = f"{scope_prefix}{name_prefix}_crop_s{split_id[0]}_{split_id[1]}"
    out = gs.Variable(
        f"{base_name}_out",
        dtype=tile.dtype,
        shape=out_shape,
    )

    node = gs.Node(
        name=base_name,
        op="Slice",
        inputs=[
            tile,
            gs.Constant(f"{base_name}_starts", np.array([rel_start_h, rel_start_w], dtype=np.int64)),
            gs.Constant(f"{base_name}_ends", np.array([rel_end_h, rel_end_w], dtype=np.int64)),
            gs.Constant(f"{base_name}_axes", np.array([axis, axis + 1], dtype=np.int64)),
            gs.Constant(f"{base_name}_steps", np.array([1, 1], dtype=np.int64)),
        ],
        outputs=[out],
    )

    return out, node



def _make_pad_constant_for_tile(original_pad_node, spatial_pads, const_name):
    """Build an ONNX Pad pads constant for one spatial tile.

    spatial_pads is (top, left, bottom, right). Non-spatial axes are kept 0
    so each tile only pads H/W enough to reproduce its exact output region.
    """
    rank = len(original_pad_node.inputs[0].shape)
    pads = np.zeros(2 * rank, dtype=np.int64)
    spatial_h = rank - 2
    spatial_w = rank - 1
    top, left, bottom, right = spatial_pads
    pads[spatial_h] = int(top)
    pads[spatial_w] = int(left)
    pads[rank + spatial_h] = int(bottom)
    pads[rank + spatial_w] = int(right)
    return gs.Constant(const_name, pads)


def _emit_tiled_node(node, split_id, input_tensors_by_index, out_range, hw_pads, name_scope):
    node_base_name = node.name or (
        node.outputs[0].name if node.outputs and node.outputs[0].name else node.op
    )
    scope_prefix = f"{name_scope}__"

    new_inputs = list(node.inputs)
    for input_index, tensor in input_tensors_by_index.items():
        new_inputs[input_index] = tensor

    assert out_range is not None, f"out_range is required when lowering op {node.op}"
    (y0, y1), (x0, x1) = out_range
    axis = _infer_spatial_axis(node.outputs[0].shape)
    out_shape = list(node.outputs[0].shape)
    out_shape[axis] = y1 - y0
    out_shape[axis + 1] = x1 - x0

    if node.op == "Reshape" and len(new_inputs) >= 2:
        shape_values = None
        if isinstance(node.inputs[1], gs.Constant):
            shape_values = np.asarray(node.inputs[1].values, dtype=np.int64).reshape(-1)

        if shape_values is not None and shape_values.size >= 2:
            target_h = out_shape[-2]
            target_w = out_shape[-1]
            if isinstance(target_h, int) and isinstance(target_w, int):
                reshaped = shape_values.copy()
                if reshaped[-2] > 0:
                    reshaped[-2] = target_h
                if reshaped[-1] > 0:
                    reshaped[-1] = target_w
                new_inputs[1] = gs.Constant(
                    f"{scope_prefix}{node_base_name}_shape_s{split_id[0]}_{split_id[1]}",
                    reshaped.astype(np.int64),
                )

    out = gs.Variable(
        f"{scope_prefix}{node.outputs[0].name}_split_s{split_id[0]}_{split_id[1]}",
        dtype=node.outputs[0].dtype,
        shape=out_shape,
    )

    attrs = dict(node.attrs) if node.attrs else {}
    if node.op in {"Conv", "AveragePool"}:
        assert hw_pads is not None, f"hw pads are required when lowering {node.op} nodes"
        attrs["pads"] = hw_pads
    elif node.op == "Pad":
        assert hw_pads is not None, "tile pads are required when lowering Pad nodes"
        assert len(new_inputs) >= 2, "Pad node must have a pads input"
        new_inputs[1] = _make_pad_constant_for_tile(
            node,
            hw_pads,
            f"{scope_prefix}{node_base_name}_pads_s{split_id[0]}_{split_id[1]}",
        )

    new_node = gs.Node(
        name=f"{scope_prefix}{node_base_name}_split_s{split_id[0]}_{split_id[1]}",
        op=node.op,
        inputs=new_inputs,
        outputs=[out],
        attrs=attrs,
    )
    return out, new_node


def _emit_group_concat(tiles, output_tensor, split_keys, tile_count, name_scope):
    split_count_h, split_count_w = tile_count
    assert len(split_keys) == len(tiles), "split_keys and tiles must have the same length"

    tile_by_key = {key: tile for key, tile in zip(split_keys, tiles)}
    axis = _infer_spatial_axis(output_tensor.shape)
    scope_prefix = f"{name_scope}__"
    concat_nodes = []

    if split_count_w == 1:
        final_concat_inputs = [tile_by_key[(split_id_h, 0)] for split_id_h in range(split_count_h)]
    else:
        final_concat_inputs = []
        for split_id_h in range(split_count_h):
            row_concat_inputs = [tile_by_key[(split_id_h, split_id_w)] for split_id_w in range(split_count_w)]
            row_out_shape = list(output_tensor.shape)
            row_out_shape[axis] = row_concat_inputs[0].shape[axis]
            row_out, row_concat_nodes = _emit_bounded_concat(
                inputs=row_concat_inputs,
                axis=axis + 1,
                output_name=f"{scope_prefix}{output_tensor.name}_row{split_id_h}",
                output_shape=row_out_shape,
                node_name=f"{scope_prefix}{output_tensor.name}_concat_row{split_id_h}",
                output_dtype=output_tensor.dtype,
            )
            concat_nodes.extend(row_concat_nodes)
            final_concat_inputs.append(row_out)

    out, final_concat_nodes = _emit_bounded_concat(
        inputs=final_concat_inputs,
        axis=axis,
        output_name=output_tensor.name,
        output_shape=output_tensor.shape,
        node_name=f"{scope_prefix}{output_tensor.name}_concat",
        output_dtype=output_tensor.dtype,
    )
    concat_nodes.extend(final_concat_nodes)

    return out, concat_nodes
