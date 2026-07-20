#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


# -----------------------------
# Output schema
# -----------------------------
@dataclass
class TSGroup:
    node_range: List[int]               # [start_node_index, end_node_index]
    tile_count: List[int]               # [split_count_h, split_count_w]
    execution_order: List[List[object]] # [[node_index, [split_id_h, split_id_w]], ...]

    def to_json_obj(self) -> dict:
        return {
            "node_range": self.node_range,
            "tile_count": self.tile_count,
            "execution_order": self.execution_order,
        }


# -----------------------------
# Helpers
# -----------------------------

def dumps_compact_lists(obj: Any, indent: int = 2) -> str:
    """
    Pretty-print JSON, but keep simple lists on ONE line:
      - [a, b]
      - [a, [b, c]]
      - [[a, [b, c]], ...]  (each item stays one line if it fits the rule)

    This is done by temporarily replacing lists with tokens, dumping,
    then substituting compact JSON for those lists back in.
    """
    token_prefix = "__LIST_TOKEN__"
    token_map = {}
    token_id = 0

    def is_compactable_list(x: Any) -> bool:
        # Compact:
        # - list of ints (e.g., [4,6])
        # - list of two ints (e.g., [0,0])
        # - list like [int, [int,int]] (e.g., [4,[0,0]])
        if not isinstance(x, list):
            return False
        if all(isinstance(v, int) for v in x):
            return True
        if len(x) == 2 and isinstance(x[0], int) and isinstance(x[1], list) \
           and len(x[1]) == 2 and all(isinstance(v, int) for v in x[1]):
            return True
        return False

    def walk(x: Any) -> Any:
        nonlocal token_id
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            # If this list is compactable, replace with token
            if is_compactable_list(x):
                tok = f"{token_prefix}{token_id}__"
                token_map[tok] = json.dumps(x, separators=(", ", ": "))
                token_id += 1
                return tok
            # Otherwise recurse
            return [walk(v) for v in x]
        return x

    cooked = walk(obj)

    # Dump with indent, but tokens are strings now
    s = json.dumps(cooked, indent=indent, ensure_ascii=False)

    # Replace "TOKEN" with compact list text (no quotes)
    for tok, compact in token_map.items():
        # match the JSON string value "TOKEN"
        s = s.replace(f"\"{tok}\"", compact)

    return s

def infer_mode_from_name(stem: str) -> Optional[str]:
    s = stem.lower()
    if "dupnas" in s:
        return "dupnas"
    if "tinyts" in s:
        return "tinyts"
    if "patchts" in s:
        return "patchts"
    return None


def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def find_related_txt(onnx_path: Path, mode: Optional[str], search_dirs: List[Path]) -> Optional[Path]:
    stem = onnx_path.stem
    patterns = {
        "dupnas": f"{stem}_pdq_config_detail_*.txt",
        "tinyts": f"{stem}_micrograph_rep_*.txt",
        "patchts": f"{stem}_tinynas_repfor2_*.txt",
    }
    if mode not in patterns:
        return None

    # dedup search dirs
    uniq_dirs = []
    seen = set()
    for d in search_dirs:
        d = d.resolve()
        if d not in seen:
            uniq_dirs.append(d)
            seen.add(d)

    candidates: List[Path] = []
    for d in uniq_dirs:
        candidates.extend(d.glob(patterns[mode]))

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# -----------------------------
# PDQ parser
# -----------------------------
def _parse_int_list(bracket_expr: str) -> List[int]:
    s = bracket_expr.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _find_first_int_list_after(label: str, text: str) -> Optional[List[int]]:
    pat = re.compile(re.escape(label) + r".*?\[(.*?)\]", re.DOTALL)
    m = pat.search(text)
    if not m:
        return None
    return _parse_int_list(m.group(1))


def _find_first_int_after(label: str, text: str) -> Optional[int]:
    pat = re.compile(re.escape(label) + r"\s*([0-9]+)")
    m = pat.search(text)
    if not m:
        return None
    return int(m.group(1))


def parse_pdq_config_detail(txt: str, *, onnx_path: Path, txt_path: Path) -> List[TSGroup]:
    useful_idx = _find_first_int_list_after("The useful subgraphs idx:", txt)
    if not useful_idx:
        return []

    blocks = re.split(r"(?=^Subgraph\s+#\d+)", txt, flags=re.MULTILINE)
    subgraph_blocks: Dict[int, str] = {}
    for b in blocks:
        m = re.match(r"^Subgraph\s+#(\d+)", b.strip())
        if m:
            subgraph_blocks[int(m.group(1))] = b

    groups: List[TSGroup] = []
    for idx in useful_idx:
        subgraph_no = idx + 1
        block = subgraph_blocks.get(subgraph_no)
        if not block:
            continue

        ops = _find_first_int_list_after("Subgraph Ops:", block)
        q = _find_first_int_after("Selected q:", block)
        d_list = _find_first_int_list_after("Selected d:", block)

        if not ops or q is None or not d_list or q <= 1:
            continue

        if "shuffle" in onnx_path.name.lower():
            if ops[0] == 0:
                start_node = ops[0]
            else:
                start_node = ops[1]
                d_list.pop(0)
        else:
            start_node = ops[0]

        end_node = ops[-1]
        execution_order: List[List[object]] = []
        for h in range(q):
            for node in d_list:
                execution_order.append([node, [h, 0]])

        groups.append(
            TSGroup(
                node_range=[start_node, end_node],
                tile_count=[q, 1],
                execution_order=execution_order,
            )
        )

    return groups


# -----------------------------
# tinyTS parser
# -----------------------------
def parse_tinyts_micrograph_rep(txt: str, *, onnx_path: Path, txt_path: Path) -> List[TSGroup]:
    micro_pat = re.compile(
        r"^Micro-Graph\s+(\d+):\s*\[(.*?)\]\s*\n(.*?)(?=^Micro-Graph\s+\d+:|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )

    peak_pat = re.compile(
        r"peak_mem_per_micro:\s*\(\s*([0-9]+)\s*,\s*(?:(None)|'([^']+)')\s*,\s*(?:(None)|([0-9]+))\s*\)",
        flags=re.MULTILINE,
    )

    dup_pat = re.compile(
        r"split input tensor num:\s*([0-9]+)",
        flags=re.MULTILINE,
    )

    def parse_ops_list(s: str) -> List[int]:
        s = s.strip()
        if not s:
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    groups: List[TSGroup] = []

    for m in micro_pat.finditer(txt):
        ops = parse_ops_list(m.group(2))
        body = m.group(3)
        if not ops:
            continue

        pm = peak_pat.search(body)
        if not pm:
            continue

        exec_type = None if pm.group(2) else (pm.group(3) or None)
        if exec_type is None:
            continue

        exec_type = exec_type.lower().strip()
        if exec_type not in ("dfs", "bfs"):
            continue

        # tile_h should come from the LAST duplication line in this micro-graph
        dup_nums = dup_pat.findall(body)
        if not dup_nums:
            continue

        tile_h = int(dup_nums[-1])
        if tile_h <= 1:
            continue

        execution_order: List[List[object]] = []
        if exec_type == "dfs":
            for h in range(tile_h):
                for node in ops:
                    execution_order.append([node, [h, 0]])
        else:  # bfs
            for node in ops:
                for h in range(tile_h):
                    execution_order.append([node, [h, 0]])

        groups.append(
            TSGroup(
                node_range=[min(ops), max(ops)],
                tile_count=[tile_h, 1],
                execution_order=execution_order,
            )
        )

    return groups


# -----------------------------
# tinyNAS parser
# -----------------------------
def parse_tinynas_repfor2(txt: str, *, onnx_path: Path, txt_path: Path) -> List[TSGroup]:
    headers = [
        "Patch selected by min latency/comp:",
        "Patch selected by min memory:",
    ]

    start_idx = -1
    for h in headers:
        i = txt.find(h)
        if i != -1:
            start_idx = i
            break
    if start_idx == -1:
        return []

    tail = txt[start_idx:]
    marks = list(re.finditer(r"^Patch selected by .*?:", tail, flags=re.MULTILINE))
    block = tail[: marks[1].start()] if len(marks) >= 2 else tail

    m_start = re.search(r"Start Op:\s*([0-9]+)\s*,\s*End Op:\s*([0-9]+)", block)
    m_q = re.search(r"Patch Split q:\s*([0-9]+)", block)
    if not m_start or not m_q:
        return []

    start_op = int(m_start.group(1))
    end_op = int(m_start.group(2))
    q = int(m_q.group(1))
    if q <= 1 or end_op < start_op:
        return []

    execution_order: List[List[object]] = []
    for h in range(q):
        for w in range(q):
            for node in range(start_op, end_op + 1):
                execution_order.append([node, [h, w]])

    return [
        TSGroup(
            node_range=[start_op, end_op],
            tile_count=[q, q],
            execution_order=execution_order,
        )
    ]


def parse_ts_groups_for_onnx(onnx_path: Path, txt_path: Optional[Path], mode: Optional[str]) -> List[TSGroup]:
    if txt_path is None or mode is None:
        return []

    txt = load_text(txt_path)
    if mode == "dupnas":
        return parse_pdq_config_detail(txt, onnx_path=onnx_path, txt_path=txt_path)
    if mode == "tinyts":
        return parse_tinyts_micrograph_rep(txt, onnx_path=onnx_path, txt_path=txt_path)
    if mode == "patchts":
        return parse_tinynas_repfor2(txt, onnx_path=onnx_path, txt_path=txt_path)
    return []


# -----------------------------
# Main: catch all ONNX + related txt, produce json
# -----------------------------
def build_config_for_dir(
    root_dir: Path,
    *,
    recursive: bool = False,
    exclude_inferred_dir: str = "inferred_onnx",
    out_dir: Optional[Path] = None,
) -> Dict[str, List[dict]]:
    root_dir = root_dir.resolve()
    out_dir = out_dir.resolve() if out_dir else None

    # collect ONNX (case-insensitive)
    if recursive:
        candidates = list(root_dir.rglob("*"))
    else:
        candidates = list(root_dir.glob("*"))

    onnx_files: List[Path] = []
    for p in candidates:
        if p.is_file() and p.suffix.lower() == ".onnx":
            # exclude inferred_onnx/*_inferred.onnx
            if exclude_inferred_dir and exclude_inferred_dir in p.parts and p.name.endswith("_inferred.onnx"):
                continue
            onnx_files.append(p)

    onnx_files.sort(key=lambda p: p.name.lower())

    results: Dict[str, List[dict]] = {}

    for onnx_path in onnx_files:
        stem = onnx_path.stem
        mode = infer_mode_from_name(stem)

        search_dirs = [onnx_path.parent, root_dir]
        txt_path = find_related_txt(onnx_path, mode, search_dirs)

        groups = parse_ts_groups_for_onnx(onnx_path, txt_path, mode)
        results[stem] = [g.to_json_obj() for g in groups]

        # write [onnxname]_config.json
        target_dir = out_dir if out_dir else onnx_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{stem}_config.json"
        out_path.write_text(dumps_compact_lists(results[stem], indent=2), encoding="utf-8")

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Directory containing ONNX + txt logs")
    ap.add_argument("--recursive", action="store_true", help="Search ONNX recursively")
    ap.add_argument("--out_dir", type=str, default="", help="Optional output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    build_config_for_dir(Path(args.root), recursive=args.recursive, out_dir=out_dir)