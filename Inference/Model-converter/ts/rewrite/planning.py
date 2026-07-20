from collections import namedtuple

from .conv import _conv_input_slice_for_output_2d, _parse_conv_spec, _parse_pool_spec

_IDENTITY_RANGE_OPS = {"Relu", "Add", "Concat", "Reshape"}

_RangePlan = namedtuple(
    "_RangePlan",
    [
        "split_keys",
        "output_ranges_by_node",
        "input_ranges_by_node",
        "hw_pads_by_node",
        "entry_ranges",
    ],
)


def _partition_ranges(total, part_count):
    base = total // part_count
    rem = total % part_count
    ranges = []
    start = 0
    for part_index in range(part_count):
        size = base + (1 if part_index < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _merge_range(existing, new_range):
    if existing is None:
        return new_range
    (y0, y1), (x0, x1) = existing
    (ny0, ny1), (nx0, nx1) = new_range
    return ((min(y0, ny0), max(y1, ny1)), (min(x0, nx0), max(x1, nx1)))


def _merge_range_list(dst_ranges, src_ranges):
    assert len(dst_ranges) == len(src_ranges)
    for idx, src_range in enumerate(src_ranges):
        dst_ranges[idx] = _merge_range(dst_ranges[idx], src_range)


def _clone_ranges(ranges):
    return [((y0, y1), (x0, x1)) for (y0, y1), (x0, x1) in ranges]


def _nonoverlap_ranges_for_tensor(output_tensor, tile_count, split_keys):
    split_count_h, split_count_w = tile_count
    height_ranges = _partition_ranges(output_tensor.shape[2], split_count_h)
    width_ranges = _partition_ranges(output_tensor.shape[3], split_count_w)
    return [
        (height_ranges[split_id_h], width_ranges[split_id_w])
        for split_id_h, split_id_w in split_keys
    ]


def _get_pad_values(node):
    """Return Pad values as (top, left, bottom, right) for NCHW tensors."""
    assert len(node.inputs) >= 2, "Pad node must have a pads input"
    pads_const = node.inputs[1]
    assert hasattr(pads_const, "values"), "Pad pads input must be a constant"

    pads = pads_const.values.reshape(-1).astype(int).tolist()
    rank = len(node.inputs[0].shape)
    assert len(pads) == 2 * rank, f"Pad pads length {len(pads)} does not match rank {rank}"

    begin = pads[:rank]
    end = pads[rank:]
    spatial_h = rank - 2
    spatial_w = rank - 1

    for axis in range(rank):
        if axis not in {spatial_h, spatial_w}:
            assert begin[axis] == 0 and end[axis] == 0, (
                "Only spatial H/W Pad is supported inside a split group; "
                f"got pads={pads}"
            )

    top = begin[spatial_h]
    left = begin[spatial_w]
    bottom = end[spatial_h]
    right = end[spatial_w]
    assert top >= 0 and left >= 0 and bottom >= 0 and right >= 0, (
        "Negative Pad/cropping is not supported inside a split group"
    )
    return top, left, bottom, right


def _pad_input_slice_for_output_2d(y0, y1, x0, x1, pad_top, pad_left, h_in, w_in):
    in_y0 = max(0, y0 - pad_top)
    in_y1 = min(h_in, y1 - pad_top)
    in_x0 = max(0, x0 - pad_left)
    in_x1 = min(w_in, x1 - pad_left)

    assert in_y0 < in_y1 and in_x0 < in_x1, (
        "Pad output tile contains only padded values; this case is not supported yet"
    )

    tile_pad_top = max(0, pad_top + in_y0 - y0)
    tile_pad_bottom = max(0, y1 - (pad_top + in_y1))
    tile_pad_left = max(0, pad_left + in_x0 - x0)
    tile_pad_right = max(0, x1 - (pad_left + in_x1))

    return (
        ((in_y0, in_y1), (in_x0, in_x1)),
        (tile_pad_top, tile_pad_left, tile_pad_bottom, tile_pad_right),
    )


def _plan_node_ranges(group_info):
    split_count_h, split_count_w = group_info.tile_count
    split_keys = [
        (split_id_h, split_id_w)
        for split_id_h in range(split_count_h)
        for split_id_w in range(split_count_w)
    ]
    split_count = len(split_keys)
    node_count = len(group_info.nodes)

    output_ranges_by_node = [[None for _ in range(split_count)] for _ in range(node_count)]
    input_ranges_by_node = [{} for _ in range(node_count)]
    hw_pads_by_node = [[] for _ in range(node_count)]
    entry_ranges = [None for _ in range(split_count)]

    # Initialize every internal sink. For normal DupNAS groups this is one node.
    # For TinyTS/PatchTS branch groups, this can be multiple branch outputs.
    for sink_local_index in group_info.sink_local_indices:
        sink_tensor = group_info.nodes[sink_local_index].outputs[0]
        output_ranges_by_node[sink_local_index] = _nonoverlap_ranges_for_tensor(
            sink_tensor,
            group_info.tile_count,
            split_keys,
        )

    # If an internal node output is consumed outside the split group, we must be
    # able to reconstruct the *full* original tensor for that outside consumer.
    # Therefore, its produced tile ranges must cover the normal non-overlapping
    # partition of the full output tensor, not only the smaller halo/demanded
    # ranges required by in-group consumers.
    #
    # Example: group 0~2, Conv_1 output input.12 is used by node 2 inside the
    # group and by nodes 4/8/11 outside the group. Node 2 may demand only 23x23
    # from each 24x24 partition, but the outside consumers need the complete
    # 48x48 input.12 reconstructed exactly.
    for external_local_index in group_info.external_output_local_indices:
        output_tensor = group_info.nodes[external_local_index].outputs[0]
        full_partition_ranges = _nonoverlap_ranges_for_tensor(
            output_tensor,
            group_info.tile_count,
            split_keys,
        )
        if all(rng is None for rng in output_ranges_by_node[external_local_index]):
            output_ranges_by_node[external_local_index] = full_partition_ranges
        else:
            _merge_range_list(
                output_ranges_by_node[external_local_index],
                full_partition_ranges,
            )

    for local_index in range(node_count - 1, -1, -1):
        node_spec = group_info.node_specs[local_index]
        node = node_spec.node

        out_ranges = output_ranges_by_node[local_index]
        if not all(rng is not None for rng in out_ranges):
            # This node is not on a path to any sink. Topology validation should
            # normally prevent this, but keep the assertion message explicit.
            assert False, f"node {node.name} has no planned output ranges"

        if node.op in {"Conv", "AveragePool"}:
            assert len(node_spec.input_sources) == 1
            main_input_index = next(iter(node_spec.input_sources))
            spec = _parse_conv_spec(node) if node.op == "Conv" else _parse_pool_spec(node)
            h_in = node.inputs[main_input_index].shape[2]
            w_in = node.inputs[main_input_index].shape[3]

            demanded_ranges = []
            hw_pads = []
            for (y0, y1), (x0, x1) in out_ranges:
                slice_info = _conv_input_slice_for_output_2d(y0, y1, x0, x1, spec, h_in, w_in)
                demanded_ranges.append(
                    (
                        (slice_info.height.slice_start, slice_info.height.slice_end),
                        (slice_info.width.slice_start, slice_info.width.slice_end),
                    )
                )
                hw_pads.append(
                    (
                        slice_info.height.pad_top,
                        slice_info.width.pad_top,
                        slice_info.height.pad_bottom,
                        slice_info.width.pad_bottom,
                    )
                )

            input_ranges_by_node[local_index] = {main_input_index: demanded_ranges}
            hw_pads_by_node[local_index] = hw_pads

        elif node.op == "Pad":
            assert len(node_spec.input_sources) == 1
            main_input_index = next(iter(node_spec.input_sources))
            pad_top, pad_left, _pad_bottom, _pad_right = _get_pad_values(node)
            h_in = node.inputs[main_input_index].shape[2]
            w_in = node.inputs[main_input_index].shape[3]

            demanded_ranges = []
            tile_pads = []
            for (y0, y1), (x0, x1) in out_ranges:
                demanded_range, tile_pad = _pad_input_slice_for_output_2d(
                    y0, y1, x0, x1, pad_top, pad_left, h_in, w_in
                )
                demanded_ranges.append(demanded_range)
                tile_pads.append(tile_pad)

            input_ranges_by_node[local_index] = {main_input_index: demanded_ranges}
            hw_pads_by_node[local_index] = tile_pads

        elif node.op in _IDENTITY_RANGE_OPS:
            input_ranges_by_node[local_index] = {
                input_index: _clone_ranges(out_ranges)
                for input_index in node_spec.input_sources
            }
            hw_pads_by_node[local_index] = [None for _ in out_ranges]
        else:
            assert False, f"unsupported op {node.op} for tiled rewrite planning"

        for input_index, demanded_ranges in input_ranges_by_node[local_index].items():
            source = node_spec.input_sources[input_index]
            if source.kind == "entry":
                _merge_range_list(entry_ranges, demanded_ranges)
            else:
                _merge_range_list(output_ranges_by_node[source.producer_local_index], demanded_ranges)

    assert all(rng is not None for rng in entry_ranges)
    for local_index, node_spec in enumerate(group_info.node_specs):
        assert all(rng is not None for rng in output_ranges_by_node[local_index])

    return _RangePlan(
        split_keys=split_keys,
        output_ranges_by_node=output_ranges_by_node,
        input_ranges_by_node=input_ranges_by_node,
        hw_pads_by_node=hw_pads_by_node,
        entry_ranges=entry_ranges,
    )
