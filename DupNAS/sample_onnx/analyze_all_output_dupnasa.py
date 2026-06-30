#!/usr/bin/env python3
"""
Analyze all DupNAS_SA outputs under one output root.

Expected output_root layout:
  output_dir/
    shufflenet/vm96/
    shufflenet/vm128/
    shufflenet/vm256/
    mobilenet/vm96/
    ...

For each model/vm folder, this script calculates:
  (1) original_feasible / total_onnx
  (2) dupnas_feasible / total_onnx
  (3) for dupnas_feasible models, the distribution of reduction_ourTS_bal
      in four bins: 1/4=[0,25), 2/4=[25,50), 3/4=[50,75), 4/4=[75,100].
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

RE_VM_DIR = re.compile(r"^vm(\d+)$", re.IGNORECASE)
RE_ORI_PEAK = re.compile(r"Peak\s+Memory\s+Usage:\s*Op\s*\d+\s*:\s*(\d+)", re.IGNORECASE)
RE_OURTS = re.compile(r"Peak\s+memory\s+for\s+all:\s*:?\s*(\d+)\s*bytes", re.IGNORECASE)

CUT_TAGS = [
    "_mem_usage",
    "_data_usage",
    "_node_info",
    "_pdq_config_detail",
    "_micrograph_rep",
    "_tinynas_repfor2",
    "_order_0_mem",
]


def base_prefix(name: str) -> str:
    name = re.sub(r"\.(txt|png|onnx|csv)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_VM\d+", "", name)
    name = re.sub(r"_goal_(bal|mem)", "", name)
    for tag in CUT_TAGS:
        if tag in name:
            return name.split(tag)[0]
    return name


def read_first_int(path: Path, regex: re.Pattern[str], max_bytes: int, max_lines: int) -> Optional[int]:
    read_bytes = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                read_bytes += len(line)
                m = regex.search(line)
                if m:
                    return int(m.group(1))
                if i >= max_lines or read_bytes >= max_bytes:
                    return None
    except OSError:
        return None
    return None


def collect_prefix_files(vm_dir: Path) -> Dict[str, List[Path]]:
    by_prefix: Dict[str, List[Path]] = defaultdict(list)
    for p in vm_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".png", ".csv"}:
            continue
        by_prefix[base_prefix(p.name)].append(p)
    return dict(by_prefix)


def pct(num: int, den: int) -> float:
    return (num / den * 100.0) if den else 0.0


def reduction_percent(ori_peak: int, after_peak: int) -> Optional[float]:
    if ori_peak <= 0 or after_peak <= 0 or after_peak > ori_peak:
        return None
    return (1.0 - after_peak / float(ori_peak)) * 100.0


def reduction_bin(red: float) -> str:
    if red < 25.0:
        return "1/4_[0,25%)"
    if red < 50.0:
        return "2/4_[25,50%)"
    if red < 75.0:
        return "3/4_[50,75%)"
    return "4/4_[75,100%]"


def find_vm_dirs(output_root: Path, model_filter: Optional[str] = None, vm_filter: Optional[int] = None) -> Iterable[Tuple[str, int, Path]]:
    """Find model/vm folders, optionally filtered by one model and one VM."""
    model_filter_norm = model_filter.lower() if model_filter else None

    for model_dir in sorted([p for p in output_root.iterdir() if p.is_dir()]):
        if model_filter_norm and model_dir.name.lower() != model_filter_norm:
            continue

        for vm_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            m = RE_VM_DIR.match(vm_dir.name)
            if not m:
                continue
            vm_kb = int(m.group(1))
            if vm_filter is not None and vm_kb != vm_filter:
                continue
            yield model_dir.name, vm_kb, vm_dir


def process_vm_dir(model: str, vm_kb: int, vm_dir: Path, args: argparse.Namespace) -> Tuple[List[dict], List[dict]]:
    vm_bytes = vm_kb * 1024
    files_by_prefix = collect_prefix_files(vm_dir)

    detail_rows: List[dict] = []
    total = 0
    original_feasible = 0
    dupnas_feasible = 0
    bins = {"1/4_[0,25%)": 0, "2/4_[25,50%)": 0, "3/4_[50,75%)": 0, "4/4_[75,100%]": 0}

    for prefix, files in sorted(files_by_prefix.items()):
        mem_file = next((p for p in files if p.name.endswith("_mem_usage.txt")), None)
        if mem_file is None:
            continue

        total += 1
        ori_peak = read_first_int(mem_file, RE_ORI_PEAK, args.max_bytes, args.max_lines) or -1
        needts = int(ori_peak > vm_bytes) if ori_peak > 0 else 0
        ori_ok = int(0 < ori_peak <= vm_bytes)
        original_feasible += ori_ok

        pdq_file = next(
            (
                p for p in files
                if "_pdq_config_detail_VM" in p.name and "goal_bal" in p.name and p.suffix.lower() == ".txt"
            ),
            None,
        )
        after_peak = read_first_int(pdq_file, RE_OURTS, args.max_bytes, args.max_lines) if pdq_file else None
        after_peak = after_peak if after_peak is not None else -1

        valid_ourTS_bal = int(needts == 1 and 0 < after_peak <= vm_bytes)
        if valid_ourTS_bal:
            dupnas_feasible += 1

        red = reduction_percent(ori_peak, after_peak) if valid_ourTS_bal else None
        red_bin = ""
        if red is not None:
            red_bin = reduction_bin(red)
            bins[red_bin] += 1

        detail_rows.append({
            "model_family": model,
            "vm": f"vm{vm_kb}",
            "onnx_name": prefix,
            "ori_peak_mem": ori_peak,
            "peak_after_ourTS_bal": after_peak,
            "needTS": needts,
            "original_feasible": ori_ok,
            "valid_ourTS_bal": valid_ourTS_bal,
            "reduction_ourTS_bal": f"{red:.2f}%" if red is not None else "0.00%",
            "reduction_quartile": red_bin,
        })

    summary_rows = [{
        "model_family": model,
        "vm": f"vm{vm_kb}",
        "total_onnx": total,
        "original_feasible": original_feasible,
        "original_feasible_ratio": f"{pct(original_feasible, total):.2f}%",
        "dupnas_feasible": dupnas_feasible,
        "dupnas_feasible_ratio": f"{pct(dupnas_feasible, total):.2f}%",
        "reduction_1/4_count": bins["1/4_[0,25%)"],
        "reduction_1/4_ratio_in_dupnasa_feasible": f"{pct(bins['1/4_[0,25%)'], dupnas_feasible):.2f}%",
        "reduction_2/4_count": bins["2/4_[25,50%)"],
        "reduction_2/4_ratio_in_dupnasa_feasible": f"{pct(bins['2/4_[25,50%)'], dupnas_feasible):.2f}%",
        "reduction_3/4_count": bins["3/4_[50,75%)"],
        "reduction_3/4_ratio_in_dupnasa_feasible": f"{pct(bins['3/4_[50,75%)'], dupnas_feasible):.2f}%",
        "reduction_4/4_count": bins["4/4_[75,100%]"],
        "reduction_4/4_ratio_in_dupnasa_feasible": f"{pct(bins['4/4_[75,100%]'], dupnas_feasible):.2f}%",
    }]
    return detail_rows, summary_rows



def print_summary_rows(rows: List[dict]) -> None:
    """Print the same summary information to console."""
    for r in rows:
        print("")
        print(f"[Summary] {r['model_family']} {r['vm']}")
        print(f"  model_family: {r['model_family']}")
        print(f"  vm: {r['vm']}")
        print(f"  total_onnx: {r['total_onnx']}")
        print(f"  original_feasible: {r['original_feasible']}")
        print(f"  original_feasible_ratio: {r['original_feasible_ratio']}")
        print(f"  dupnas_feasible: {r['dupnas_feasible']}")
        print(f"  dupnas_feasible_ratio: {r['dupnas_feasible_ratio']}")
        print(f"  reduction_1/4_count: {r['reduction_1/4_count']}")
        print(f"  reduction_1/4_ratio_in_dupnasa_feasible: {r['reduction_1/4_ratio_in_dupnasa_feasible']}")
        print(f"  reduction_2/4_count: {r['reduction_2/4_count']}")
        print(f"  reduction_2/4_ratio_in_dupnasa_feasible: {r['reduction_2/4_ratio_in_dupnasa_feasible']}")
        print(f"  reduction_3/4_count: {r['reduction_3/4_count']}")
        print(f"  reduction_3/4_ratio_in_dupnasa_feasible: {r['reduction_3/4_ratio_in_dupnasa_feasible']}")
        print(f"  reduction_4/4_count: {r['reduction_4/4_count']}")
        print(f"  reduction_4/4_ratio_in_dupnasa_feasible: {r['reduction_4/4_ratio_in_dupnasa_feasible']}")

def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        print(f"[WARN] No rows for {path.name}; skip writing")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="Root output directory containing model/vm folders")
    ap.add_argument("--model", choices=["shufflenet", "mobilenet", "inception"],
                    help="Analyze only one model family, e.g., shufflenet")
    ap.add_argument("--vm", type=int, choices=[96, 128, 256],
                    help="Analyze only one VM setting in KB, e.g., 96")
    ap.add_argument("--summary_csv", default=None,
                    help="Output summary CSV name. Default: all_output_dupnasa_summary.csv, or <model>_vm<vm>_summary.csv when filters are used.")
    ap.add_argument("--detail_csv", default=None,
                    help="Output detail CSV name. Default: all_output_dupnasa_detail.csv, or <model>_vm<vm>_detail.csv when filters are used.")
    ap.add_argument("--max_bytes", type=int, default=8 * 1024 * 1024)
    ap.add_argument("--max_lines", type=int, default=20000)
    args = ap.parse_args()

    output_root = Path(args.output_dir).resolve()
    if not output_root.is_dir():
        raise SystemExit(f"[ERROR] output_dir does not exist: {output_root}")

    all_details: List[dict] = []
    all_summaries: List[dict] = []

    for model, vm_kb, vm_dir in find_vm_dirs(output_root, args.model, args.vm):
        print(f"[Info] Process {model}/vm{vm_kb}: {vm_dir}")
        detail_rows, summary_rows = process_vm_dir(model, vm_kb, vm_dir, args)
        all_details.extend(detail_rows)
        all_summaries.extend(summary_rows)

    if not all_summaries:
        filt = []
        if args.model:
            filt.append(f"model={args.model}")
        if args.vm:
            filt.append(f"vm={args.vm}")
        filt_msg = " with " + ", ".join(filt) if filt else ""
        raise SystemExit(f"[ERROR] No model/vm output folders found{filt_msg}, or no *_mem_usage.txt files detected")

    if args.summary_csv is None:
        if args.model and args.vm:
            args.summary_csv = f"{args.model}_vm{args.vm}_summary.csv"
        else:
            args.summary_csv = "all_output_dupnasa_summary.csv"

    if args.detail_csv is None:
        if args.model and args.vm:
            args.detail_csv = f"{args.model}_vm{args.vm}_detail.csv"
        else:
            args.detail_csv = "all_output_dupnasa_detail.csv"

    print_summary_rows(all_summaries)

    write_csv(output_root / args.summary_csv, all_summaries)
    write_csv(output_root / args.detail_csv, all_details)


if __name__ == "__main__":
    main()
