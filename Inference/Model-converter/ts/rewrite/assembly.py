from .lowering import _emit_entry_tiles, _emit_group_concat, _emit_tile_crop, _emit_tiled_node
from .planning import _plan_node_ranges


def _build_group(group_info):
    range_plan = _plan_node_ranges(group_info)
    split_count = len(range_plan.split_keys)

    group_start, group_end = group_info.node_range
    name_scope = f"g{group_start}_{group_end}"

    # Split the group input once, then reuse tiles throughout the rewrite.
    entry_tiles, entry_slice_nodes = _emit_entry_tiles(
        group_info.entry_tensor,
        range_plan.entry_ranges,
        name_scope,
    )

    split_pos_by_key = {split_key: split_pos for split_pos, split_key in enumerate(range_plan.split_keys)}

    tiles_by_local_index = [[None for _ in range(split_count)] for _ in group_info.nodes]
    body_nodes = []

    # Lower each requested (node, tile) step in execution order.
    for orig_index, split_id in group_info.execution_order:
        split_pos = split_pos_by_key[split_id]
        local_index = orig_index - group_start
        assert 0 <= local_index < len(group_info.nodes)

        node_spec = group_info.node_specs[local_index]
        demanded_ranges = range_plan.input_ranges_by_node[local_index]

        input_tensors_by_index = {}

        for input_index, source in node_spec.input_sources.items():
            if source.kind == "entry":
                source_tile = entry_tiles[split_pos]
                produced_range = range_plan.entry_ranges[split_pos]
            else:
                source_tile = tiles_by_local_index[source.producer_local_index][split_pos]
                produced_range = range_plan.output_ranges_by_node[source.producer_local_index][split_pos]
            assert source_tile is not None

            if input_index in demanded_ranges:
                required_range = demanded_ranges[input_index][split_pos]
                if produced_range != required_range:
                    source_tile, crop_node = _emit_tile_crop(
                        source_tile,
                        produced_range,
                        required_range,
                        split_id,
                        f"{node_spec.node.name}_l{local_index}_in{input_index}",
                        name_scope,
                    )
                    body_nodes.append(crop_node)

            input_tensors_by_index[input_index] = source_tile

        output_tile, lowered_node = _emit_tiled_node(
            node_spec.node,
            split_id,
            input_tensors_by_index,
            range_plan.output_ranges_by_node[local_index][split_pos],
            range_plan.hw_pads_by_node[local_index][split_pos],
            name_scope,
        )
        tiles_by_local_index[local_index][split_pos] = output_tile
        body_nodes.append(lowered_node)

    for local_index, node_tiles in enumerate(tiles_by_local_index):
        assert all(tile is not None for tile in node_tiles)

    stitched_exit, concat_nodes = _emit_group_concat(
        tiles_by_local_index[-1],
        group_info.exit_tensor,
        range_plan.split_keys,
        group_info.tile_count,
        name_scope,
    )

    ordered_nodes = entry_slice_nodes + body_nodes + concat_nodes
    return ordered_nodes, stitched_exit


def _apply_group(graph, orig_nodes, group_info, new_nodes, stitched_exit):
    node_a = orig_nodes[group_info.node_range[0]]
    node_b = orig_nodes[group_info.node_range[1]]

    start_pos = next(i for i, node in enumerate(graph.nodes) if node is node_a)
    end_pos = next(i for i, node in enumerate(graph.nodes) if node is node_b)

    for consumer in list(group_info.exit_tensor.outputs):
        for idx, inp in enumerate(consumer.inputs):
            if inp is group_info.exit_tensor:
                consumer.inputs[idx] = stitched_exit

    for idx, out in enumerate(graph.outputs):
        if out is group_info.exit_tensor:
            graph.outputs[idx] = stitched_exit

    for node in group_info.nodes:
        node.inputs = []
        node.outputs = []

    graph.nodes = graph.nodes[:start_pos] + new_nodes + graph.nodes[end_pos + 1 :]
