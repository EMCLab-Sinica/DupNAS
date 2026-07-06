from .lowering import _emit_entry_tiles, _emit_group_concat, _emit_tile_crop, _emit_tiled_node
from .planning import _partition_ranges, _plan_node_ranges


def _build_group(group_info):
    range_plan = _plan_node_ranges(group_info)
    split_count = len(range_plan.split_keys)

    group_start, group_end = group_info.node_range
    name_scope = f"g{group_start}_{group_end}"

    entry_tiles, entry_slice_nodes = _emit_entry_tiles(
        group_info.entry_tensor,
        range_plan.entry_ranges,
        name_scope,
    )

    split_pos_by_key = {split_key: split_pos for split_pos, split_key in enumerate(range_plan.split_keys)}
    tiles_by_local_index = [[None for _ in range(split_count)] for _ in group_info.nodes]
    body_nodes = []

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

    stitched_outputs = []
    concat_nodes_all = []

    def _nonoverlap_ranges_for_tensor(output_tensor):
        split_count_h, split_count_w = group_info.tile_count
        height_ranges = _partition_ranges(output_tensor.shape[2], split_count_h)
        width_ranges = _partition_ranges(output_tensor.shape[3], split_count_w)
        return [
            (height_ranges[split_id_h], width_ranges[split_id_w])
            for split_id_h, split_id_w in range_plan.split_keys
        ]

    def _tiles_for_full_reconstruction(local_index, output_tensor):
        produced_ranges = range_plan.output_ranges_by_node[local_index]
        target_ranges = _nonoverlap_ranges_for_tensor(output_tensor)
        reconstructed_tiles = []

        for split_pos, (tile, produced_range, target_range) in enumerate(
            zip(tiles_by_local_index[local_index], produced_ranges, target_ranges)
        ):
            split_id = range_plan.split_keys[split_pos]
            if produced_range != target_range:
                tile, crop_node = _emit_tile_crop(
                    tile,
                    produced_range,
                    target_range,
                    split_id,
                    f"{output_tensor.name}_full_reconstruct",
                    name_scope,
                )
                concat_nodes_all.append(crop_node)
            reconstructed_tiles.append(tile)

        return reconstructed_tiles

    # Reconstruct every output that is consumed outside the group. This supports
    # both the original single-sink case and TinyTS/PatchTS multi-sink branch
    # groups that stop before the merge/Concat.
    seen_tensors = set()
    for local_index in group_info.external_output_local_indices:
        output_tensor = group_info.nodes[local_index].outputs[0]
        if id(output_tensor) in seen_tensors:
            continue
        seen_tensors.add(id(output_tensor))

        reconstruction_tiles = _tiles_for_full_reconstruction(local_index, output_tensor)
        stitched_output, concat_nodes = _emit_group_concat(
            reconstruction_tiles,
            output_tensor,
            range_plan.split_keys,
            group_info.tile_count,
            name_scope,
        )
        stitched_outputs.append((output_tensor, stitched_output))
        concat_nodes_all.extend(concat_nodes)

    ordered_nodes = entry_slice_nodes + body_nodes + concat_nodes_all
    return ordered_nodes, stitched_outputs


def _apply_group(graph, orig_nodes, group_info, new_nodes, stitched_outputs):
    node_a = orig_nodes[group_info.node_range[0]]
    node_b = orig_nodes[group_info.node_range[1]]

    start_pos = next(i for i, node in enumerate(graph.nodes) if node is node_a)
    end_pos = next(i for i, node in enumerate(graph.nodes) if node is node_b)

    for original_tensor, stitched_tensor in stitched_outputs:
        for consumer in list(original_tensor.outputs):
            for idx, inp in enumerate(consumer.inputs):
                if inp is original_tensor:
                    consumer.inputs[idx] = stitched_tensor

        for idx, out in enumerate(graph.outputs):
            if out is original_tensor:
                graph.outputs[idx] = stitched_tensor

    for node in group_info.nodes:
        node.inputs = []
        node.outputs = []

    graph.nodes = graph.nodes[:start_pos] + new_nodes + graph.nodes[end_pos + 1 :]
