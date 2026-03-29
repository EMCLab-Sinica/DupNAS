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

    height_ranges = _partition_ranges(group_info.exit_tensor.shape[2], split_count_h)
    width_ranges = _partition_ranges(group_info.exit_tensor.shape[3], split_count_w)
    output_ranges_by_node[-1] = [
        (h_range, w_range) for h_range in height_ranges for w_range in width_ranges
    ]

    for local_index in range(node_count - 1, -1, -1):
        node_spec = group_info.node_specs[local_index]
        node = node_spec.node

        out_ranges = output_ranges_by_node[local_index]
        assert all(rng is not None for rng in out_ranges)

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
