# ONNX Tensor Splitter

This tool rewrites ONNX models to execute selected node ranges tile-by-tile. For convolution chains, it adds explicit boundary-overlap logic to keep results numerically equivalent to the original model.

## What this tool does

Given an ONNX model and a split config, this tool generates a new ONNX model where selected node ranges run tile-by-tile instead of on full tensors.

For each configured node range, it:

1. Splits the range input tensor into `tile_count` tiles.
2. Computes each tile's required Conv input range so results match the original full-tensor run.
3. Rewrites operators in that range to consume and produce per-tile tensors.
4. Stitches tile outputs back into the original tensor layout.

After rewriting, the CLI verifies numerical equivalence with ONNX Runtime.

## Quick start

1. Create and activate a virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Create a split config JSON file (see [Configuration format](#configuration-format)) and run the rewrite command.

   ```bash
   python -m ts.cli input.onnx config.json output.onnx
   ```

4. Check the verification result (`PASS` or `FAIL`) printed by the CLI.

## CLI reference

### Usage

```bash
python -m ts.cli INPUT CONFIG OUTPUT
```

### Arguments

| Name | Description |
| --- | --- |
| `INPUT` | Path to input ONNX model. |
| `CONFIG` | Path to split configuration JSON. |
| `OUTPUT` | Path to output ONNX model. |

## Configuration format

The config must be a JSON list of group entries.

Each group contains:

- `node_range`: `[start_node_index, end_node_index]`
- `tile_count`: `[split_count_h, split_count_w]`
- `execution_order`: list of `[node_index, [split_id_h, split_id_w]]`

### Example

```json
[
  {
    "node_range": [4, 6],
    "tile_count": [2, 1],
    "execution_order": [
      [4, [0, 0]],
      [4, [1, 0]],
      [5, [0, 0]],
      [5, [1, 0]],
      [6, [0, 0]],
      [6, [1, 0]]
    ]
  },
  {
    "node_range": [10, 11],
    "tile_count": [2, 2],
    "execution_order": [
      [10, [0, 0]],
      [11, [0, 0]],
      [10, [0, 1]],
      [11, [0, 1]],
      [10, [1, 0]],
      [11, [1, 0]],
      [10, [1, 1]],
      [11, [1, 1]]
    ]
  }
]
```

Notes:

- `node_range` is inclusive.
- `execution_order` must include each pair in the group exactly once.

## Testing

Run the test suite:

```bash
pytest
```
