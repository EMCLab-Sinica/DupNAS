#!/usr/bin/env python3
"""
Collect summary metrics from DupNAS ALLSEEDS random-sample reports only.

This script intentionally ignores per-seed files such as:
  *_combo_random_samples_VM*_seed*.txt

It only reads ALLSEEDS files such as:
  *_combo_random_samples_ALLSEEDS_VM*.txt

Report items extracted per model/VM:
  - TOTAL SAMPLES
  - TOTAL TIME
  - AVG TIME
  - Overall feasible samples satisfy VM
  - Overall best sampled peak, shown as peak_after_ts

Extra aggregate/console report:
  - number of models whose Overall feasible samples satisfy VM > 0
  - average Overall feasible samples satisfy VM among those models only
  - DupNAS-excluded feasible network candidates (%)
  - Feasible configuration sets (%)

Example:
  python collect_allseed_report_summary.py --input_dir s10xr1000_invalid_EXGboth_VM96 --recursive

Output CSV name inferred from --input_dir:
  allseed_report_summary_EXGboth_VM96.csv
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ALLSEED_FILE_RE = re.compile(
    r"^(?P<model>.+?)_combo_random_samples_ALLSEEDS_VM(?P<vm>\d+)\.txt$"
)

PEAK_RE = re.compile(
    r"peak_after_ts\s*=\s*(?P<peak_b>\d+)\s*B\s*\((?P<peak_kb>[0-9.]+)\s*KB\)",
    flags=re.IGNORECASE,
)

SAMPLED_LINE_RE = re.compile(
    r"Sampled combo\s+(?P<sample_id>\d+):.*?"
    r"peak_after_ts\s*=\s*(?P<peak_b>\d+)\s*B\s*\((?P<peak_kb>[0-9.]+)\s*KB\).*?"
    r"satisfy_vm\s*=\s*(?P<satisfy_vm>True|False)",
    flags=re.IGNORECASE,
)


def safe_int(x):
    try:
        if x is None or str(x).strip() == "":
            return np.nan
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return np.nan


def safe_float(x):
    try:
        if x is None or str(x).strip() == "":
            return np.nan
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan


def model_family(model_name):
    s = str(model_name).lower()
    if "mbv2" in s or "mobilenet" in s:
        return "mbv2"
    if "shuffle" in s:
        return "shuffle"
    if "incept" in s or "inception" in s:
        return "incept"
    return s


def infer_meta_from_filename(path: Path):
    m = ALLSEED_FILE_RE.match(path.name)
    if m:
        return m.group("model"), int(m.group("vm"))

    vm = np.nan
    vm_m = re.search(r"VM(?P<vm>\d+)", path.name, flags=re.IGNORECASE)
    if vm_m:
        vm = int(vm_m.group("vm"))

    model = re.sub(r"_combo_random_samples_ALLSEEDS.*$", "", path.name)
    model = re.sub(r"\.txt$", "", model)
    return model, vm


def find_number_after_label(text, label_patterns):
    """Return the first numeric value after any label pattern."""
    for label in label_patterns:
        m = re.search(
            label + r"\s*[:=]\s*(?P<value>[0-9.eE+\-,]+)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group("value")
    return None


def parse_count_total_after_label(text, label_patterns):
    """Parse labels like 'Overall feasible samples satisfy VM: 4974/10000'."""
    for label in label_patterns:
        m = re.search(
            label
            + r"\s*[:=]\s*(?P<count>[0-9,]+)\s*/\s*(?P<total>[0-9,]+)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            return safe_int(m.group("count")), safe_int(m.group("total"))

        # Also support labels followed by only a count.
        m = re.search(
            label + r"\s*[:=]\s*(?P<count>[0-9,]+)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            return safe_int(m.group("count")), np.nan

    return np.nan, np.nan


def parse_overall_best_peak(text):
    """
    Extract peak_after_ts from the 'Overall best sampled peak' block.
    If the block is not present, fall back to the lowest feasible sampled-line peak
    inside the ALLSEEDS file.
    """
    # Preferred: explicit overall best sampled peak block.
    m = re.search(
        r"Overall\s+best\s+sampled\s+peak.*?(?P<body>peak_after_ts\s*=\s*\d+\s*B\s*\([0-9.]+\s*KB\))",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        pm = PEAK_RE.search(m.group("body"))
        if pm:
            return safe_float(pm.group("peak_b")), safe_float(pm.group("peak_kb")), "overall_best_sampled_peak"

    # Some reports put peak_after_ts on the next line after the title.
    m = re.search(
        r"Overall\s+best\s+sampled\s+peak(?P<body>.*?)(?:\n\n|PER-SEED SUMMARY|Seed\s+\d+|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        pm = PEAK_RE.search(m.group("body"))
        if pm:
            return safe_float(pm.group("peak_b")), safe_float(pm.group("peak_kb")), "overall_best_sampled_peak"

    # Fallback: compute min peak among feasible sampled combo lines from the ALLSEEDS file.
    best_b = None
    best_kb = None
    for line in text.splitlines():
        sm = SAMPLED_LINE_RE.search(line)
        if not sm:
            continue
        if sm.group("satisfy_vm").lower() != "true":
            continue
        peak_b = safe_float(sm.group("peak_b"))
        peak_kb = safe_float(sm.group("peak_kb"))
        if best_b is None or peak_b < best_b:
            best_b = peak_b
            best_kb = peak_kb

    if best_b is not None:
        return best_b, best_kb, "computed_min_feasible_sampled_line"

    return np.nan, np.nan, "not_found"


def parse_allseed_report(path: Path):
    model, vm = infer_meta_from_filename(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    total_samples = safe_int(find_number_after_label(text, [r"TOTAL\s+SAMPLES"]))
    total_time_sec = safe_float(find_number_after_label(text, [r"TOTAL\s+TIME"]))
    avg_time_sec = safe_float(find_number_after_label(text, [r"AVG\s+TIME", r"AVERAGE\s+TIME"]))

    feasible_count, feasible_total = parse_count_total_after_label(
        text,
        [
            r"Overall\s+feasible\s+samples\s+satisfy\s+VM",
            r"Overall\s+feasible\s+samples\s+satisfying\s+VM",
            r"Overall\s+feasible",
        ],
    )

    # If the feasible line includes a denominator, use it as a fallback total_samples.
    if pd.isna(total_samples) and not pd.isna(feasible_total):
        total_samples = feasible_total

    if pd.isna(avg_time_sec) and not pd.isna(total_time_sec) and not pd.isna(total_samples) and total_samples > 0:
        avg_time_sec = total_time_sec / total_samples

    best_peak_b, best_peak_kb, best_peak_source = parse_overall_best_peak(text)

    return {
        "source_file": str(path),
        "model": model,
        "model_family": model_family(model),
        "vm": vm,
        "total_samples": total_samples,
        "total_time_sec": total_time_sec,
        "avg_time_sec": avg_time_sec,
        "overall_feasible_samples_satisfy_vm": feasible_count,
        "overall_feasible_samples_total": feasible_total,
        "overall_feasible_samples_satisfy_vm_ratio": (
            feasible_count / total_samples
            if not pd.isna(feasible_count) and not pd.isna(total_samples) and total_samples > 0
            else np.nan
        ),
        "overall_best_sampled_peak_after_ts_B": best_peak_b,
        "overall_best_sampled_peak_after_ts_KB": best_peak_kb,
        "overall_best_sampled_peak_source": best_peak_source,
    }



EXGRULE_ALIAS = {
    "both": "neither",
    "boundary": "PConly",
    "branches": "BPonly",
    "neither": "neither",
    "pconly": "PConly",
    "bponly": "BPonly",
}

MODEL_ALIAS = {
    "shuffle": "shufflenet",
    "shufflenet": "shufflenet",
    "incept": "inception",
    "inception": "inception",
    "mbv2": "mobilenet",
    "mobilenet": "mobilenet",
}


def normalize_exgrule(exgrule: Optional[str]):
    if exgrule is None:
        return None
    key = str(exgrule).strip()
    return EXGRULE_ALIAS.get(key.lower(), key)


def normalize_model(model: Optional[str]):
    if model is None:
        return None
    key = str(model).strip()
    return MODEL_ALIAS.get(key.lower(), key)


def build_input_dir(root: Path, exgrule: Optional[str], model: Optional[str], vm_setting: Optional[int]):
    """
    New output layout from the run scripts:
      HEtest/outputs/<EXGRULE>/<model>/vm<VM_SETTING>
    """
    if exgrule is None and model is None and vm_setting is None:
        return root

    missing = []
    if exgrule is None:
        missing.append("--exgrule")
    if model is None:
        missing.append("--model")
    if vm_setting is None:
        missing.append("--vm_setting")
    if missing:
        raise ValueError(
            "When using the new HEtest layout, provide all of: " + ", ".join(missing)
        )

    return root / "outputs" / normalize_exgrule(exgrule) / normalize_model(model) / f"vm{vm_setting}"


def infer_output_tag(input_dir: Path, exgrule: Optional[str] = None, model: Optional[str] = None, vm_setting: Optional[int] = None):
    """Infer a compact output suffix for the summary CSV."""
    if exgrule is not None or model is not None or vm_setting is not None:
        parts = []
        if exgrule is not None:
            parts.append(normalize_exgrule(exgrule))
        if model is not None:
            parts.append(normalize_model(model))
        if vm_setting is not None:
            parts.append(f"VM{vm_setting}")
        return "_".join(parts)

    # Fallback for old flat folders, e.g. s10xr1000_invalid_EXGboth_VM96.
    name = Path(input_dir).name
    exg = None
    vm = None

    exg_m = re.search(
        r"(EXG(?:both|boundary|branches|one)|neither|PConly|BPonly)",
        name,
        flags=re.IGNORECASE,
    )
    if exg_m:
        raw = exg_m.group(1)
        exg_map = {
            "exgboth": "EXGboth",
            "exgboundary": "EXGboundary",
            "exgbranches": "EXGbranches",
            "exgone": "EXGone",
            "neither": "neither",
            "pconly": "PConly",
            "bponly": "BPonly",
        }
        exg = exg_map.get(raw.lower(), raw)

    vm_m = re.search(r"(?:^|[_/-])VM?(?P<vm>\d+)(?:[_/-]|$)", name, flags=re.IGNORECASE)
    if vm_m:
        vm = f"VM{vm_m.group('vm')}"

    if exg and vm:
        return f"{exg}_{vm}"
    if exg:
        return exg
    if vm:
        return vm
    return None


def default_output_csv_name(input_dir: Path, exgrule: Optional[str] = None, model: Optional[str] = None, vm_setting: Optional[int] = None):
    tag = infer_output_tag(input_dir, exgrule, model, vm_setting)
    if tag:
        return f"allseed_report_summary_{tag}.csv"
    return "allseed_report_summary.csv"



def compute_exclusion_console_metrics(df: pd.DataFrame, model_group=None):
    """
    Console metrics for one <exgrule, model, vm_setting> run.

    DupNAS-excluded feasible network candidates (%):
      number of originally infeasible models whose excluded configurations contain
      at least one VM-feasible configuration / total originally infeasible models.

    Feasible configuration sets (%):
      average feasible-configuration ratio among only the above feasible models.
    """
    feasible_col = "overall_feasible_samples_satisfy_vm"
    total_col = "total_samples"

    parsed_models = int(len(df))

    # Total originally-infeasible model candidates should use the full candidate count,
    # not only the number of ALLSEEDS files currently parsed.
    # shuffle/shufflenet has 2400 candidates; mobilenet/inception have 3200.
    m = str(model_group or "").lower()
    if m in {"shuffle", "shufflenet"}:
        total_models = 2400
    elif m in {"mobilenet", "mbv2", "inception", "incept"}:
        total_models = 3200
    else:
        total_models = parsed_models

    feasible_counts = pd.to_numeric(df[feasible_col], errors="coerce").fillna(0)
    total_samples = pd.to_numeric(df[total_col], errors="coerce")

    positive_mask = feasible_counts > 0
    feasible_models = int(positive_mask.sum())

    dupnas_excluded_feasible_pct = (
        feasible_models / total_models * 100.0 if total_models > 0 else 0.0
    )

    ratios = feasible_counts[positive_mask] / total_samples[positive_mask]
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
    feasible_config_sets_pct = float(ratios.mean() * 100.0) if len(ratios) else 0.0

    return {
        "parsed_allseed_models": parsed_models,
        "total_models": total_models,
        "dupnas_excluded_feasible_models": feasible_models,
        "dupnas_excluded_feasible_network_candidates_pct": dupnas_excluded_feasible_pct,
        "feasible_configuration_sets_pct": feasible_config_sets_pct,
    }

def write_report_with_top_section(df: pd.DataFrame, out_path: Path):
    feasible_col = "overall_feasible_samples_satisfy_vm"
    positive = df[pd.to_numeric(df[feasible_col], errors="coerce").fillna(0) > 0]

    # Count ALLSEEDS files / model rows.
    num_allseed_files = int(len(df))

    # Count unique models. If one model may appear with multiple VM settings,
    # this avoids double-counting the same model.
    num_unique_models = int(df["model"].nunique())

    console_metrics = compute_exclusion_console_metrics(df, model_group=df["model_group"].iloc[0] if "model_group" in df.columns and len(df) else None)

    aggregate = pd.DataFrame([
        {
            "metric": "number_of_allseed_files",
            "value": num_allseed_files,
        },
        {
            "metric": "total_originally_infeasible_models_for_percentage",
            "value": console_metrics["total_models"],
        },
        {
            "metric": "number_of_unique_models",
            "value": num_unique_models,
        },
        {
            "metric": "models_with_overall_feasible_samples_satisfy_vm_gt_0",
            "value": int(len(positive)),
        },
        {
            "metric": "avg_overall_feasible_samples_satisfy_vm_among_gt_0_models",
            "value": float(positive[feasible_col].mean()) if len(positive) else 0.0,
        },
        {
            "metric": "dupnas_excluded_feasible_network_candidates_pct",
            "value": console_metrics["dupnas_excluded_feasible_network_candidates_pct"],
        },
        {
            "metric": "feasible_configuration_sets_pct",
            "value": console_metrics["feasible_configuration_sets_pct"],
        },
    ])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write("# Aggregate report\n")
        aggregate.to_csv(f, index=False)
        f.write("\n")
        f.write("# Per-model ALLSEEDS report\n")
        df.to_csv(f, index=False)

    return aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory. When running inside HEtest, keep the default '.'. New layout reads from <root>/outputs/<EXGRULE>/<model>/vm<VM_SETTING>.",
    )
    ap.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Optional direct input directory. If omitted, it is built from --root, --exgrule, --model, and --vm_setting.",
    )
    ap.add_argument(
        "--exgrule",
        type=str,
        default=None,
        choices=["neither", "PConly", "BPonly", "both", "boundary", "branches"],
        help="EXGRULE folder name. Old names are accepted and mapped: both->neither, boundary->PConly, branches->BPonly.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["shufflenet", "inception", "mobilenet", "shuffle", "incept", "mbv2"],
        help="Model folder name under HEtest/outputs/<EXGRULE>/.",
    )
    ap.add_argument(
        "--vm_setting",
        type=int,
        default=None,
        help="VM setting used in folder name vm<VM_SETTING>.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory. Default: HEtest, so summary CSV is saved under HEtest/.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*_combo_random_samples_ALLSEEDS_VM*.txt",
        help="Glob pattern. Keep this as ALLSEEDS only unless your filename format differs.",
    )
    ap.add_argument("--recursive", action="store_true", help="Search input_dir recursively.")
    ap.add_argument("--filter_vm", type=int, default=None, help="Only collect one VM size. Usually unnecessary when --vm_setting is set.")
    ap.add_argument("--filter_model", type=str, default=None, help="Only collect models containing this substring.")
    ap.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV filename. If omitted, use allseed_report_summary_<EXGRULE>_<model>_VM<VM>.csv.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    exgrule = normalize_exgrule(args.exgrule)
    model = normalize_model(args.model)

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = build_input_dir(root, exgrule, model, args.vm_setting)
        # Convenience fallback: if the user runs this script from inside HEtest
        # but an older script/default passed --root HEtest, avoid looking for
        # HEtest/outputs inside HEtest.
        if not input_dir.exists() and root.name == "HEtest":
            alt_input_dir = build_input_dir(Path("."), exgrule, model, args.vm_setting)
            if alt_input_dir.exists():
                input_dir = alt_input_dir

    outdir = Path(args.outdir) if args.outdir else root
    if not outdir.exists() and outdir.name == "HEtest" and Path("outputs").exists():
        outdir = Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    globber = input_dir.rglob if args.recursive else input_dir.glob
    files = sorted(set(globber(args.pattern)))

    # Hard guard: do not accidentally read per-seed files.
    files = [f for f in files if "_combo_random_samples_ALLSEEDS_VM" in f.name]

    # If --vm_setting is given, use it as the default VM filter.
    filter_vm = args.filter_vm if args.filter_vm is not None else args.vm_setting
    if filter_vm is not None:
        files = [f for f in files if f"VM{filter_vm}" in f.name]

    # Do not filter by --model here, because the model is already encoded in the
    # folder path outputs/<EXGRULE>/<model>/vm<VM>. The ALLSEEDS filenames may not
    # contain strings like "shufflenet" or "mobilenet". Only apply this filter
    # when the user explicitly provides --filter_model.
    if args.filter_model:
        files = [f for f in files if args.filter_model.lower() in f.name.lower()]

    if not files:
        print(f"[ERROR] No ALLSEEDS files found in: {input_dir}")
        print("        Check that you run from HEtest, or pass --root /4TB/aeuser/DupNAS-AE/DupNAS/HEtest.")
        print("        Also note: --model is used for the folder path only, not as a filename filter.")
        return 1

    rows = []
    for path in files:
        try:
            row = parse_allseed_report(path)
            row["exgrule"] = exgrule if exgrule is not None else ""
            row["model_group"] = model if model is not None else ""
            rows.append(row)
            print(f"[READ] {path}")
        except Exception as e:
            print(f"[WARN] Skip {path}: {e}")

    if not rows:
        print("[ERROR] No ALLSEEDS reports could be parsed.")
        return 1

    df = pd.DataFrame(rows)
    df = df.sort_values(["model_family", "model", "vm"], na_position="last").reset_index(drop=True)

    output_csv = args.output_csv or default_output_csv_name(input_dir, exgrule, model, args.vm_setting)
    out_csv = outdir / output_csv
    aggregate = write_report_with_top_section(df, out_csv)

    # Also save a clean per-model CSV without the top text sections for easy pandas reading.
    clean_name = "allseed_report_by_model_vm_clean.csv"
    if exgrule is not None or model is not None or args.vm_setting is not None:
        clean_tag = infer_output_tag(input_dir, exgrule, model, args.vm_setting)
        clean_name = f"allseed_report_by_model_vm_clean_{clean_tag}.csv"
    clean_csv = outdir / clean_name
    #df.to_csv(clean_csv, index=False)

    console_metrics = compute_exclusion_console_metrics(df, model_group=model)

    current_exgrule = exgrule if exgrule is not None else "all"
    current_model = model if model is not None else "all"
    current_vm = args.vm_setting if args.vm_setting is not None else "all"

    print(f"[INPUT] {input_dir}")
    print(f"[SAVE] {out_csv}")
    #print(f"[SAVE] {clean_csv}")
    print("[CONSOLE SUMMARY]")
    print(f"  Current setting: exgrule={current_exgrule}, model={current_model}, vm_setting={current_vm}")
    print(
        "  DupNAS-excluded feasible network candidates (%): "
        f"{console_metrics['dupnas_excluded_feasible_network_candidates_pct']:.4f}% "
        f"({console_metrics['dupnas_excluded_feasible_models']}/{console_metrics['total_models']})"
    )
    print(
        "  Feasible configuration sets (%): "
        f"{console_metrics['feasible_configuration_sets_pct']:.4f}%"
    )
    print("[SUMMARY]")
    for _, r in aggregate.iterrows():
        print(f"  {r['metric']}: {r['value']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
