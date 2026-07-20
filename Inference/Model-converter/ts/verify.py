import numpy as np
import onnx
import onnxruntime as ort

DEFAULT_VERIFY_RTOL = 1e-4
DEFAULT_VERIFY_ATOL = 1e-5


def _extract_static_shape(name, value_info):
    """Return a fully static input shape, asserting verification preconditions."""
    tensor_type = value_info.type.tensor_type
    shape = [
        dim.dim_value if dim.HasField("dim_value") else None
        for dim in tensor_type.shape.dim
    ]
    assert not any(dim == 0 for dim in shape if dim is not None), (
        f"input {name} has invalid dimension 0"
    )
    assert not any(dim is None for dim in shape), (
        f"input {name} must have static shapes for verification"
    )
    return tuple(int(dim) for dim in shape)


def _resolve_numpy_dtype(name, value_info):
    tensor_type = value_info.type.tensor_type
    # dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(tensor_type.elem_type)
    try:
        from onnx import mapping
        dtype = mapping.TENSOR_TYPE_TO_NP_TYPE.get(tensor_type.elem_type)
    except Exception:
        from onnx.helper import tensor_dtype_to_np_dtype
        dtype = tensor_dtype_to_np_dtype(tensor_type.elem_type)
    assert dtype is not None, f"input {name} has unsupported dtype {tensor_type.elem_type}"
    return dtype


def _make_input_array(name, value_info, rng):
    static_shape = _extract_static_shape(name, value_info)
    dtype = _resolve_numpy_dtype(name, value_info)
    return rng.standard_normal(size=static_shape).astype(dtype)


def _build_inference_session(model, sess_options):
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options,
        providers=["CPUExecutionProvider"],
    )


def _collect_runtime_inputs(model, rng):
    initializer_names = {init.name for init in model.graph.initializer}
    return {
        inp.name: _make_input_array(inp.name, inp, rng)
        for inp in model.graph.input
        if inp.name not in initializer_names
    }


def _compute_output_diffs(original, orig_outs, new_outs, rtol, atol):
    """Compute per-output max abs diffs and aggregate allclose pass/fail."""
    assert len(orig_outs) == len(new_outs), (
        "output count mismatch between original and rewritten models"
    )

    diffs = {}
    ok = True
    for output_info, orig, new in zip(original.graph.output, orig_outs, new_outs):
        name = output_info.name
        if orig.shape != new.shape:
            diffs[name] = float("inf")
            ok = False
            continue

        diff = float(np.max(np.abs(orig - new)))
        diffs[name] = diff
        if not np.allclose(orig, new, rtol=rtol, atol=atol):
            ok = False

    return ok, diffs


def verify_model(
    original,
    rewritten,
    rtol=DEFAULT_VERIFY_RTOL,
    atol=DEFAULT_VERIFY_ATOL,
):
    """Run both models in ONNX Runtime and compare outputs elementwise."""
    sess_options = ort.SessionOptions()
    sess_original = _build_inference_session(original, sess_options)
    sess_rewritten = _build_inference_session(rewritten, sess_options)

    rng = np.random.default_rng(0)
    inputs = _collect_runtime_inputs(original, rng)
    for name, value in inputs.items():
        print(f"Verification input {name}: shape={value.shape}, dtype={value.dtype}")

    orig_outs = sess_original.run(None, inputs)
    new_outs = sess_rewritten.run(None, inputs)

    return _compute_output_diffs(original, orig_outs, new_outs, rtol=rtol, atol=atol)
