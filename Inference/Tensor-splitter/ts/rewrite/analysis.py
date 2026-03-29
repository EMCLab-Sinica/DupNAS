from collections import namedtuple

import onnx_graphsurgeon as gs

SUPPORTED_GROUP_OPS = {"Conv", "Relu", "Add", "Concat", "AveragePool", "Reshape"}

_InputSource = namedtuple(
    "_InputSource",
    ["kind", "producer_local_index"],
)

_NodeSpec = namedtuple(
    "_NodeSpec",
    ["node", "local_index", "input_sources"],
)

_GroupInfo = namedtuple(
    "_GroupInfo",
    [
        "node_range",
        "tile_count",
        "execution_order",
        "nodes",
        "entry_tensor",
        "exit_tensor",
        "node_specs",
    ],
)


def _ensure_supported_op(node):
    assert node.op in SUPPORTED_GROUP_OPS, f"unsupported op in split group: {node.op}"
    if node.op == "Concat":
        assert node.attrs["axis"] == 1, "Concat in split group must use axis=1"


def _build_adjacency(node_count, internal_edges):
    out_edges = [[] for _ in range(node_count)]
    in_edges = [[] for _ in range(node_count)]
    for src, dst in internal_edges:
        out_edges[src].append(dst)
        in_edges[dst].append(src)
    return in_edges, out_edges


def _reachable_indices(seed_indices, edges):
    visited = set()
    stack = list(seed_indices)
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(edges[current])
    return visited


def _validate_group_topology(node_specs, in_edges, out_edges):
    source_local_indices = [index for index, incoming in enumerate(in_edges) if not incoming]
    assert source_local_indices, "group must have at least one source node"

    sink_local_indices = [index for index, outgoing in enumerate(out_edges) if not outgoing]
    assert len(sink_local_indices) == 1, "group must have exactly one sink node"
    sink_local_index = sink_local_indices[0]
    assert sink_local_index == len(node_specs) - 1, "group sink node must be the last node in node_range"

    join_local_indices = [index for index, incoming in enumerate(in_edges) if len(incoming) >= 2]
    if join_local_indices:
        assert len(join_local_indices) == 1, "group may have at most one join node"
        assert join_local_indices[0] == sink_local_index, "join node must be the sink node"

    reachable_from_sources = _reachable_indices(source_local_indices, out_edges)
    reachable_to_sink = _reachable_indices([sink_local_index], in_edges)
    for spec in node_specs:
        assert spec.local_index in reachable_from_sources, f"node {spec.node.name} is not reachable from any group source node"
        assert spec.local_index in reachable_to_sink, f"node {spec.node.name} does not feed the group sink node"

    return sink_local_index


def _validate_non_sink_outputs(group_nodes, sink_local_index, graph_outputs):
    group_node_ids = {id(node) for node in group_nodes}
    graph_output_ids = {id(tensor) for tensor in graph_outputs}

    for local_index, node in enumerate(group_nodes):
        if local_index == sink_local_index:
            continue

        output_tensor = node.outputs[0]
        external_consumers = [
            consumer for consumer in output_tensor.outputs if id(consumer) not in group_node_ids
        ]

        assert not external_consumers, (
            f"only the sink node may have external consumers, but {node.name} feeds "
            f"{external_consumers[0].name}"
        )
        assert id(output_tensor) not in graph_output_ids, (
            "only the sink node may feed graph outputs, but non-sink node "
            f"{node.name} produces graph output tensor {output_tensor.name}"
        )


def _collect_node_specs(group_nodes, graph_outputs):
    local_index_by_id = {id(node): local_index for local_index, node in enumerate(group_nodes)}

    entry_tensor = None
    internal_edges = []
    node_specs = []

    for local_index, node in enumerate(group_nodes):
        _ensure_supported_op(node)
        input_sources = {}

        for input_index, tensor in enumerate(node.inputs):
            if isinstance(tensor, gs.Constant):
                continue

            producers = tensor.inputs
            assert len(producers) <= 1, f"tensor {tensor.name} must have at most one producer"

            producer_local_index = None
            if producers:
                producer_local_index = local_index_by_id.get(id(producers[0]))

            if producer_local_index is None:
                if entry_tensor is None:
                    entry_tensor = tensor
                else:
                    assert tensor is entry_tensor, "group source nodes must all read the same external entry tensor"
                input_sources[input_index] = _InputSource(
                    kind="entry",
                    producer_local_index=None,
                )
            else:
                input_sources[input_index] = _InputSource(
                    kind="node",
                    producer_local_index=producer_local_index,
                )
                internal_edges.append((producer_local_index, local_index))

        assert input_sources, f"node {node.name} must have at least one input"
        assert len(node.outputs) == 1, f"node {node.name} must have exactly one output"

        node_specs.append(
            _NodeSpec(
                node=node,
                local_index=local_index,
                input_sources=input_sources,
            )
        )

    assert entry_tensor is not None, "group must have exactly one external entry tensor"

    in_edges, out_edges = _build_adjacency(len(group_nodes), internal_edges)
    sink_local_index = _validate_group_topology(node_specs, in_edges, out_edges)
    _validate_non_sink_outputs(group_nodes, sink_local_index, graph_outputs)

    return entry_tensor, node_specs


def _analyze_group(orig_nodes, group_cfg, graph_outputs):
    start, end = group_cfg.node_range
    assert 0 <= start <= end < len(orig_nodes), f"invalid node_range {group_cfg.node_range}"

    group_nodes = orig_nodes[start : end + 1]
    entry_tensor, node_specs = _collect_node_specs(group_nodes, graph_outputs)
    exit_tensor = group_nodes[-1].outputs[0]

    return _GroupInfo(
        node_range=group_cfg.node_range,
        tile_count=group_cfg.tile_count,
        execution_order=group_cfg.execution_order,
        nodes=group_nodes,
        entry_tensor=entry_tensor,
        exit_tensor=exit_tensor,
        node_specs=node_specs,
    )
