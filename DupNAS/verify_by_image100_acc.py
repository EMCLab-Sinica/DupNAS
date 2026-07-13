#!/usr/bin/env python3
"""
Validate ONNX models on the same ImageNet-100 validation dataset pipeline used
by DupNAS fine-tuning.

Key difference from verify_by_image100_acc_rank0.py:
  - This script imports NASBase.model.common_utils.get_dataset(...)
    instead of manually loading images with PIL.
  - It uses the val_loader.dataset returned by get_dataset(), so preprocessing,
    class order, and dataset ordering should match fine-tuning.
  - It applies a rank shard equivalent to DistributedSampler(shuffle=False),
    e.g., rank=0/world_size=4 to match rank-0 DDP validation logs.

Typical use from /4TB/aeuser/DupNAS-AE/DupNAS:

python3.9 verify_by_image100_acc_rank0_getdataset.py \
  --onnx-dirs genonnx/shuffle \
  --output genonnx/onnx_image100_accuracy_shuffle_rank0_getdataset.csv \
  --batch-size 1 \
  --provider cpu \
  --ddp-rank 0 \
  --ddp-world-size 4

The output CSV contains only the final summary columns:
  model, top1_accuracy, top5_accuracy
"""

import argparse
import csv
import os
import re
import sys
import contextlib
import io
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is required. Install by: pip install onnxruntime or onnxruntime-gpu"
    ) from e

try:
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
except ImportError as e:
    raise ImportError("PyTorch is required because this script reuses get_dataset().") from e


DEFAULT_ROOT = Path(__file__).resolve().parent
MODEL_DIRS = [
    DEFAULT_ROOT / "genonnx/shuffle",
    DEFAULT_ROOT / "genonnx/mbv2",
    DEFAULT_ROOT / "genonnx/incept",
]



def create_ort_session(onnx_path: Path, providers):
    """
    Create an ONNX Runtime session with graph optimizations disabled.

    Inception models were verified to match PyTorch numerically only when
    ORT graph optimization was disabled. The exported ONNX graph itself is
    correct; the default optimized ORT execution changes the final result.
    """
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )

    return ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=providers,
    )


def parse_ir_from_name(path: Path):
    """Example: shuffle-im100-vm96-dupnas_w0.5_ir128.onnx -> 128"""
    m = re.search(r"_ir(\d+)", path.stem)
    return int(m.group(1)) if m else None


def get_input_info(session, onnx_path: Path):
    inp = session.get_inputs()[0]
    name = inp.name
    shape = inp.shape

    layout = "NCHW"
    input_size = None
    fixed_batch = None

    if len(shape) == 4:
        s = shape
        fixed_batch = s[0] if isinstance(s[0], int) else None

        if s[1] == 3:
            layout = "NCHW"
            if isinstance(s[2], int):
                input_size = s[2]
        elif s[3] == 3:
            layout = "NHWC"
            if isinstance(s[1], int):
                input_size = s[1]

    if input_size is None:
        input_size = parse_ir_from_name(onnx_path)

    if input_size is None:
        raise ValueError(
            f"Cannot infer input resolution for {onnx_path.name}. "
            f"ONNX input shape is {shape}. Please pass --input-size."
        )

    return name, shape, layout, input_size, fixed_batch


def softmax_or_logits_topk(output, topk=5):
    """ONNX output is usually logits. Top-k is unchanged by softmax."""
    output = np.asarray(output)
    if output.ndim > 2:
        output = output.reshape(output.shape[0], -1)

    k = min(topk, output.shape[1])
    topk_idx = np.argpartition(-output, kth=k - 1, axis=1)[:, :k]

    row_indices = np.arange(output.shape[0])[:, None]
    topk_scores = output[row_indices, topk_idx]
    order = np.argsort(-topk_scores, axis=1)
    topk_idx = topk_idx[row_indices, order]
    return topk_idx


def collect_onnx_files(model_dirs):
    files = []
    for d in model_dirs:
        if not d.exists():
            print(f"[WARN] Missing model dir: {d}")
            continue
        files.extend(sorted(d.glob("*.onnx")))
    return files


def ensure_project_importable(root: Path):
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_settings_and_get_dataset(root: Path):
    ensure_project_importable(root)

    # Import after sys.path is prepared.
    from settings import Settings
    from NASBase.model.common_utils import get_dataset

    # Make sure the intended dataset is used, matching your fine-tuning logs.
    Settings.NAS_SETTINGS_GENERAL["DATASET"] = "IMAGE100"
    return Settings, get_dataset


def get_rank_shard_loader(dataset, batch_size: int, ddp_rank: int, ddp_world_size: int, num_workers: int):
    """
    Match PyTorch DistributedSampler(shuffle=False, drop_last=False).
    This is the validation sampling behavior used in train_supernet.py under DDP.
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=False,
        drop_last=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def get_val_dataset_for_input_size(Settings, get_dataset, input_size: int, finetune_batch_size: int):
    """
    Reuse the exact get_dataset(...) path used by fine-tuning:
      _, input_resolution = model.net_choices
      orig_train_loader, orig_val_loader = get_dataset(..., input_resolution=input_resolution,
                                                       trainset_batchsize=per_gpu_bs)

    get_dataset() prints many details. Capture them and only keep the key dataset-loaded line
    so the validation log stays compact.
    """
    print(f"[INFO] Loading get_dataset() validation dataset for input_size={input_size}")

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        _train_loader, val_loader = get_dataset(
            Settings,
            input_resolution=input_size,
            trainset_batchsize=finetune_batch_size,
        )

    for line in captured.getvalue().splitlines():
        if "Dataset:" in line and "has already loaded" in line:
            print(line)

    return val_loader.dataset


def tensor_to_numpy_for_onnx(inputs, layout: str):
    if torch.is_tensor(inputs):
        x = inputs.detach().cpu().numpy().astype(np.float32)
    else:
        x = np.asarray(inputs, dtype=np.float32)

    # get_dataset() should return NCHW tensors. Convert only if ONNX requires NHWC.
    if layout == "NHWC" and x.ndim == 4 and x.shape[1] == 3:
        x = np.transpose(x, (0, 2, 3, 1))

    return x.astype(np.float32, copy=False)


def labels_to_numpy(labels):
    if torch.is_tensor(labels):
        return labels.detach().cpu().numpy().astype(np.int64)
    return np.asarray(labels, dtype=np.int64)


def evaluate_onnx_on_loader(
    onnx_path: Path,
    loader,
    batch_size: int,
    provider: str,
    forced_input_size=None,
):
    available = ort.get_available_providers()
    if provider == "cuda" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = create_ort_session(onnx_path, providers)
    input_name, input_shape, layout, inferred_size, fixed_batch = get_input_info(session, onnx_path)
    input_size = forced_input_size if forced_input_size is not None else inferred_size
    output_names = [o.name for o in session.get_outputs()]

    effective_batch_size = batch_size
    if fixed_batch == 1 and batch_size != 1:
        print(
            f"[WARN] {onnx_path.name} has fixed batch size 1. "
            f"Using batch_size=1 instead of {batch_size}."
        )
        effective_batch_size = 1

    # If the caller passed a loader with larger batch size, split inside the loop.
    total = 0
    top1_correct = 0
    top5_correct = 0

    for inputs, labels in loader:
        labels_np_all = labels_to_numpy(labels)
        x_all = tensor_to_numpy_for_onnx(inputs, layout=layout)

        for start in range(0, x_all.shape[0], effective_batch_size):
            x = x_all[start : start + effective_batch_size]
            labels_np = labels_np_all[start : start + effective_batch_size]

            # Safety for fixed-batch models: skip incomplete chunk only if fixed batch > 1.
            if fixed_batch is not None and fixed_batch != x.shape[0]:
                if fixed_batch == 1:
                    # Should not happen because effective_batch_size=1.
                    pass
                else:
                    raise ValueError(
                        f"{onnx_path.name} expects fixed batch {fixed_batch}, "
                        f"but got chunk batch {x.shape[0]}. Use --batch-size {fixed_batch}."
                    )

            outputs = session.run(output_names, {input_name: x})
            logits = outputs[0]
            topk_idx = softmax_or_logits_topk(logits, topk=5)

            pred1 = topk_idx[:, 0]
            top1_correct += int(np.sum(pred1 == labels_np))

            for i, label in enumerate(labels_np):
                if label in topk_idx[i]:
                    top5_correct += 1

            total += len(labels_np)

    top1 = top1_correct / total * 100.0 if total else 0.0
    top5 = top5_correct / total * 100.0 if total else 0.0

    return {
        "onnx": str(onnx_path),
        "model_name": onnx_path.name,
        "input_shape": str(input_shape),
        "layout": layout,
        "input_size": input_size,
        "num_images": total,
        "top1_acc": top1,
        "top5_acc": top5,
        "provider": providers[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="DupNAS root directory. Default: directory containing this script.",
    )
    parser.add_argument(
        "--onnx-dirs",
        type=Path,
        nargs="*",
        default=MODEL_DIRS,
        help="Folders containing ONNX models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "onnx_image100_accuracy_rank0_getdataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="ONNX inference batch size. Use 1 for fixed-batch ONNX models.")
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Use cuda if onnxruntime-gpu is installed.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Force one input image size for all ONNX models. Normally omit this and infer from each ONNX/filename.",
    )
    parser.add_argument(
        "--ddp-rank",
        type=int,
        default=0,
        help="DDP rank to evaluate. Use 0 to match rank-0 fine-tuning logs.",
    )
    parser.add_argument(
        "--ddp-world-size",
        type=int,
        default=4,
        help="DDP world size used during fine-tuning, e.g., 4.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for the rank-shard validation loader.",
    )
    parser.add_argument(
        "--finetune-batch-size",
        type=int,
        default=200,
        help="Batch size passed to get_dataset(), matching IMAGE100 FINETUNE_BATCHSIZE in your log.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit after DDP sharding. Do not use for matching fine-tuning accuracy.",
    )
    args = parser.parse_args()

    if args.ddp_world_size <= 0:
        raise ValueError("--ddp-world-size must be > 0")
    if not (0 <= args.ddp_rank < args.ddp_world_size):
        raise ValueError("--ddp-rank must satisfy 0 <= rank < world_size")

    root = args.root.resolve()
    Settings, get_dataset = load_settings_and_get_dataset(root)

    onnx_files = collect_onnx_files(args.onnx_dirs)
    if not onnx_files:
        raise RuntimeError("No ONNX files found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the CSV consistent with the final printed summary only.
    fieldnames = [
        "model",
        "top1_accuracy",
        "top5_accuracy",
    ]

    dataset_cache = {}
    summary_rows = []

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, onnx_path in enumerate(onnx_files, start=1):
            print(f"[{i}/{len(onnx_files)}] Evaluating: {onnx_path}")

            try:
                # Open once to infer input size before constructing get_dataset().
                tmp_session = create_ort_session(onnx_path, ["CPUExecutionProvider"])
                _input_name, _input_shape, _layout, inferred_size, _fixed_batch = get_input_info(tmp_session, onnx_path)
                del tmp_session

                input_size = args.input_size if args.input_size is not None else inferred_size

                if input_size not in dataset_cache:
                    dataset = get_val_dataset_for_input_size(
                        Settings=Settings,
                        get_dataset=get_dataset,
                        input_size=input_size,
                        finetune_batch_size=args.finetune_batch_size,
                    )
                    dataset_cache[input_size] = dataset

                dataset = dataset_cache[input_size]
                loader = get_rank_shard_loader(
                    dataset=dataset,
                    batch_size=args.batch_size,
                    ddp_rank=args.ddp_rank,
                    ddp_world_size=args.ddp_world_size,
                    num_workers=args.num_workers,
                )

                if args.limit is not None:
                    # For debugging only. This is after rank sharding.
                    from torch.utils.data import Subset
                    sampler_indices = list(loader.sampler)
                    limited_indices = sampler_indices[: args.limit]
                    limited_dataset = Subset(dataset, limited_indices)
                    loader = DataLoader(
                        limited_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=False,
                    )
                    print(f"[WARN] Limit enabled after DDP sharding: {args.limit} images")

                result = evaluate_onnx_on_loader(
                    onnx_path=onnx_path,
                    loader=loader,
                    batch_size=args.batch_size,
                    provider=args.provider,
                    forced_input_size=input_size,
                )

                result.update(
                    {
                        "ddp_rank": args.ddp_rank,
                        "ddp_world_size": args.ddp_world_size,
                        "dataset_source": "NASBase.model.common_utils.get_dataset",
                    }
                )

                summary_row = {
                    "model": result["model_name"],
                    "top1_accuracy": f"{result['top1_acc']:.2f}%",
                    "top5_accuracy": f"{result['top5_acc']:.2f}%",
                }
                writer.writerow(summary_row)
                f.flush()
                summary_rows.append(result)

                print(
                    f"[RESULT] {onnx_path.name}: "
                    f"Top-1={result['top1_acc']:.2f}%, "
                    # f"Top-5={result['top5_acc']:.2f}%"
                )
                print()

            except Exception as e:
                print(f"[ERROR] Failed on {onnx_path.name}: {e}")
                writer.writerow(
                    {
                        "model": onnx_path.name,
                        "top1_accuracy": "ERROR",
                        # "top5_accuracy": "ERROR",
                    }
                )
                f.flush()

    if summary_rows:
        print("Accuracy summary:")
        print(f"{'model':<60} {'Top-1 accuracy':>15}" ) # {'Top-5 accuracy':>15}")
        for row in summary_rows:
            print(
                f"{row['model_name']:<60} "
                f"{row['top1_acc']:>14.2f}% "
                # f"{row['top5_acc']:>14.2f}%"
            )

    print(f"[DONE] Saved accuracy results to: {args.output}")


if __name__ == "__main__":
    # Avoid accidentally letting torch distributed env from a shell affect this script.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    main()
