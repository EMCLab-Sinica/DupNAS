import sys
import os.path
import itertools
import copy
from pprint import pprint
#from .train_supernet import run_supernet_train
import statistics
import traceback
from typing import Dict, List, Tuple
import numpy as np
import torch
import onnx
import onnxruntime as ort
import re, ast
import csv


from settings import Settings, SSOptPolicy, Stages, arg_parser
from NASBase import file_utils, utils
#from NASBase.evo_search.search import evo_search
from NASBase.model.common_utils import get_supernet, parametric_supernet_choices, parametric_supernet_blk_choices
#from NASBase.ss_optimization.ss_opt import ss_optimization
#from NASBase.fine_tune import fine_tune_best_solution
#from logger.remote_logger import get_remote_logger_obj, get_remote_logger_basic_init_params
from NASBase.ss_optimization.subnet_utils import sample_subnet_configs_from_file, check_constraints, merge_constraint_stats

from NASBase.hw_cost.Modules_nas_v1.IEExplorer.plat_perf import PlatPerf
from NASBase.model.common_utils import get_network_dimension, get_network_obj, netobj_to_pyobj, get_supernet, get_dummy_net_input_tensor



if Settings.NAS_SETTINGS_GENERAL['ARC'] == 'mbv2':
    from NASBase.model.mbv2_arch import MNASSubNet, MNASSubNet
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'shuffle':
    from NASBase.model.shuffle_arch import MNASSubNet, MNASSubNet
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'incept':
    from NASBase.model.inception_arch import MNASSubNet, MNASSubNet


supernet_choices, _ = parametric_supernet_choices(global_settings=Settings)
supernet_block_choices = parametric_supernet_blk_choices(global_settings=Settings)

dataset = Settings.NAS_SETTINGS_GENERAL['DATASET']
first_block_hard_coded=Settings.NAS_SETTINGS_PER_DATASET[dataset]['FIRST_BLOCK_HARD_CODED']


from collections import defaultdict


#SPEC_FILE = os.path.join(os.path.dirname(__file__), "spec_models_.txt")
SPEC_FILE = os.path.join(os.path.dirname(__file__), "spec_models_"+Settings.NAS_SETTINGS_GENERAL['ARC']+".txt")


def build_finetuned_subnet_from_supernet(supernet, subnet_config):
    """
    Build the requested subnet architecture, then replace its trainable modules
    with the matching fine-tuned modules from the loaded supernet.
    """
    subnet = MNASSubNet(**subnet_config)

    subnet.stem = copy.deepcopy(supernet.stem)
    subnet.classifier = copy.deepcopy(supernet.classifier)

    for bix, block_choice in enumerate(subnet.choice_per_block):
        try:
            choice_idx = supernet.blk_choices.index(list(block_choice))
        except ValueError as exc:
            raise ValueError(
                f"Block {bix} choice {block_choice} is not in the supernet search space"
            ) from exc
        subnet.choice_blocks[bix] = copy.deepcopy(supernet.choice_blocks[bix][choice_idx])

    return subnet

def parse_spec_file(path: str):
    """
    Returns: dict[(wm: float, ir: int)] -> list[(name: str, cpb: list-of-lists)]
    """
    if not os.path.isfile(path):
        print(f"[ERROR] Spec file not found: {os.path.abspath(path)}")
        return {}

    groups = defaultdict(list)
    auto_id = 0

    # very forgiving parse: key=value, comma separated
    # supports keys: wm|width_multiplier, ir|input_resolution, name|subnet, cpb
    for raw in open(path, "r", encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # split top-level commas only (no commas inside cpb lists)
        # simple approach: find 'cpb=' and slice it out
        kv_part = line
        cpb_text = None
        m_cpb = re.search(r"\bcpb\s*=\s*(\[\[.*\]\])\s*$", line)
        if m_cpb:
            cpb_text = m_cpb.group(1)
            kv_part = line[:m_cpb.start()].rstrip(", ")

        wm = None
        ir = None
        name = None

        # parse the k=v pairs before cpb
        for piece in [p.strip() for p in kv_part.split(",") if p.strip()]:
            if "=" not in piece:
                continue
            k, v = [x.strip() for x in piece.split("=", 1)]
            k_low = k.lower()
            if k_low in ("wm", "width_multiplier"):
                wm = float(v)
            elif k_low in ("ir", "input_resolution"):
                ir = int(v)
            elif k_low in ("name", "subnet"):
                name = v

        if wm is None or ir is None or cpb_text is None:
            print(f"[WARN] Skip line (need wm/ir/cpb): {line}")
            continue

        try:
            cpb = ast.literal_eval(cpb_text)
        except Exception as e:
            print(f"[WARN] Bad CPB on line: {line}\n  -> {e}")
            continue

        if name is None:
            name = f"spec{auto_id}"
            auto_id += 1

        groups[(wm, ir)].append((name, cpb))

    return dict(groups)


# ======= SPEC MODE (replaces the for-loop over supernet_choices) =======
# ======= SPEC MODE (replaces the for-loop over supernet_choices) =======
spec_groups = parse_spec_file(SPEC_FILE)

# collect summary rows for one CSV across all (wm, ir) groups
summary_rows = []

if not spec_groups:
    print("[INFO] No specs found; nothing to export.")
else:

    performance_model = PlatPerf(Settings.NAS_SETTINGS_GENERAL, Settings.PLATFORM_SETTINGS)

    for (width_multiplier, input_resolution), cpb_tuples in sorted(spec_groups.items()):
        print(f"\n[INFO] Processing specs for wm={width_multiplier}, ir={input_resolution} "
              f"({len(cpb_tuples)} subnet(s))")

        net_input = get_dummy_net_input_tensor(Settings, input_resolution)
        onnx_path = Settings.NAS_EVOSEARCH_SETTINGS['GEN_ONNX_FILE_PATH']
        ckpt_path = Settings.NAS_SETTINGS_GENERAL['CHECKPOINT_DIR']
        os.makedirs(onnx_path, exist_ok=True)

        for idx, (name, cpb) in enumerate(cpb_tuples):
            # Only export/check DupNAS models for now.
            if "dupnas" not in name.lower():
                print(f"[SKIP] Non-DupNAS model: {name}")
                continue

            try:
                ckptname = f"{name}_supernet_{Settings.NAS_SETTINGS_GENERAL['ARC']}_best-fine-tuned.pth"
                print(f"[INFO] Loading subnet-specific supernet: {ckptname}")

                ckpt_file = os.path.join(ckpt_path, ckptname)
                supernet = get_supernet(
                    global_settings=Settings,
                    dataset=dataset,
                    load_state=True,
                    supernet_train_chkpnt_fname=ckpt_file,
                    width_multiplier=width_multiplier
                )
                supernet.eval()

                # only generate config for this subnet
                subcfg_iter = sample_subnet_configs_from_file(
                    supernet,
                    [(name, cpb)],
                    first_block_hard_coded=first_block_hard_coded
                )
                subcfgs = list(subcfg_iter)

                if not subcfgs:
                    print(f"[WARN] No valid subnet produced for name={name}")
                    summary_rows.append({
                        "wm": width_multiplier,
                        "ir": input_resolution,
                        "name": name,
                        "cpb": cpb,
                        "max_features": 0,
                        "flops_sum": 0,
                    })
                    continue

                each_subnet_config = subcfgs[0]
                each_subnet = build_finetuned_subnet_from_supernet(
                    supernet,
                    each_subnet_config,
                )
                each_subnet.eval()
                subnet_name = getattr(each_subnet, "name", name)

                # ---------------------------------------------------------
                # CHECK 1: fine-tuned supernet path vs extracted subnet
                # ---------------------------------------------------------
                choice_ixs = [
                    supernet.blk_choices.index(list(block_choice))
                    for block_choice in each_subnet.choice_per_block
                ]

                print("[CHECK] Requested CPB:", cpb)
                print("[CHECK] Extracted subnet CPB:", each_subnet.choice_per_block)
                print("[CHECK] Supernet choice indexes:", choice_ixs)

                supernet_device = next(supernet.parameters()).device
                check_input = net_input.to(supernet_device)

                supernet.eval()
                each_subnet = each_subnet.to(supernet_device)
                each_subnet.eval()

                with torch.no_grad():
                    supernet_output = supernet(check_input, choice_ixs)
                    subnet_output = each_subnet(check_input)

                supernet_output_cpu = supernet_output.detach().cpu()
                subnet_output_cpu = subnet_output.detach().cpu()

                supernet_subnet_abs_diff = (
                    supernet_output_cpu - subnet_output_cpu
                ).abs()

                supernet_subnet_max_diff = (
                    supernet_subnet_abs_diff.max().item()
                )
                supernet_subnet_mean_diff = (
                    supernet_subnet_abs_diff.mean().item()
                )

                print(
                    "[CHECK] Supernet vs subnet max abs diff: "
                    f"{supernet_subnet_max_diff:.10f}"
                )
                print(
                    "[CHECK] Supernet vs subnet mean abs diff: "
                    f"{supernet_subnet_mean_diff:.10f}"
                )

                if supernet_subnet_max_diff > 1e-5:
                    raise RuntimeError(
                        "Extracted subnet does not numerically match the "
                        "fine-tuned supernet path. "
                        f"max_abs_diff={supernet_subnet_max_diff:.10f}, "
                        f"choice_ixs={choice_ixs}, cpb={cpb}"
                    )

                print("[CHECK] Supernet and extracted subnet match.")

                print(f"[INFO] Exporting fine-tuned subnet: wm={width_multiplier}, ir={input_resolution}, name={subnet_name}")

                subnet_dims = get_network_dimension(each_subnet, input_tensor=net_input)
                subnet_obj = get_network_obj(subnet_dims)

                all_layers_fit_nvm, network_nvm_usage, _ = performance_model.get_nvm_usage(subnet_obj)

                print("len of network_nvm_usage: ", len(network_nvm_usage))
                max_features = max((f + w for f, w in network_nvm_usage), default=0)

                network_flops, _, _ = performance_model.get_network_flops(
                    subnet_obj, fixed_params=None, layer_based_cals=True
                )
                flops_sum = int(sum(network_flops)) if network_flops else 0

                model_name = f"{subnet_name}_w{width_multiplier}_ir{input_resolution}"
                onnx_file = os.path.join(onnx_path, model_name + ".onnx")

                # Export from CPU to avoid device-dependent export behavior.
                each_subnet = each_subnet.cpu()
                export_input = net_input.detach().cpu()

                # Family-specific ONNX export settings:
                #   ShuffleNet / MobileNetV2: opset 11 + constant folding
                #   Inception:                opset 13 + no constant folding
                model_family = Settings.NAS_SETTINGS_GENERAL["ARC"].lower()

                if model_family in {"shuffle", "mbv2"}:
                    export_opset = 11
                    export_constant_folding = True
                elif model_family == "incept":
                    export_opset = 11
                    export_constant_folding = True
                else:
                    raise ValueError(
                        f"Unsupported model family for ONNX export: {model_family}"
                    )

                torch.onnx.export(
                    each_subnet,
                    export_input,
                    onnx_file,
                    input_names=["input"],
                    output_names=["output"],
                    export_params=True,
                    opset_version=export_opset,
                    do_constant_folding=export_constant_folding,
                    training=torch.onnx.TrainingMode.EVAL,
                    # dynamic_axes={
                    #     "input": {0: "batch_size"},
                    #     "output": {0: "batch_size"},
                    # },
                )

                # Recompute the PyTorch reference after export. Some exporters
                # may temporarily alter module state during tracing.
                each_subnet.eval()
                with torch.no_grad():
                    pytorch_export_output = (
                        each_subnet(export_input)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                if not os.path.exists(onnx_file) or os.path.getsize(onnx_file) < 10_000:
                    print(f"[*] ONNX too small or missing: {onnx_file}")
                    summary_rows.append({
                        "wm": width_multiplier,
                        "ir": input_resolution,
                        "name": subnet_name,
                        "cpb": cpb,
                        "max_features": max_features,
                        "flops_sum": flops_sum,
                    })
                    continue

                try:
                    onnx_model = onnx.load(onnx_file)
                    onnx.checker.check_model(onnx_model)

                    # -----------------------------------------------------
                    # CHECK 2: extracted PyTorch subnet vs exported ONNX
                    # -----------------------------------------------------
                    ort_session = ort.InferenceSession(
                        onnx_file,
                        providers=["CPUExecutionProvider"],
                    )
                    ort_input_name = ort_session.get_inputs()[0].name

                    onnx_output = ort_session.run(
                        None,
                        {
                            ort_input_name: export_input.numpy().astype(
                                np.float32,
                                copy=False,
                            )
                        },
                    )[0]

                    onnx_abs_diff = np.abs(
                        pytorch_export_output - onnx_output
                    )
                    subnet_onnx_max_diff = float(onnx_abs_diff.max())
                    subnet_onnx_mean_diff = float(onnx_abs_diff.mean())

                    print(
                        "[CHECK] Subnet vs ONNX max abs diff: "
                        f"{subnet_onnx_max_diff:.10f}"
                    )
                    print(
                        "[CHECK] Subnet vs ONNX mean abs diff: "
                        f"{subnet_onnx_mean_diff:.10f}"
                    )

                    if subnet_onnx_max_diff > 1e-4:
                        raise RuntimeError(
                            "Exported ONNX output does not numerically match "
                            "the extracted PyTorch subnet. "
                            f"max_abs_diff={subnet_onnx_max_diff:.10f}, "
                            f"onnx={onnx_file}"
                        )

                    print("[CHECK] Extracted subnet and ONNX match.")

                except Exception as e:
                    print(f"[*] ONNX check failed: {onnx_file}\n    {e}")
                    summary_rows.append({
                        "wm": width_multiplier,
                        "ir": input_resolution,
                        "name": subnet_name,
                        "cpb": cpb,
                        "max_features": max_features,
                        "flops_sum": flops_sum,
                    })
                    continue

                print(f"[OK] {onnx_file}")

                summary_rows.append({
                    "wm": width_multiplier,
                    "ir": input_resolution,
                    "name": subnet_name,
                    "cpb": cpb,
                    "max_features": max_features,
                    "flops_sum": flops_sum,
                })

            except Exception as e:
                print(f"[ERR] Export failed for subnet {name} (wm={width_multiplier}, ir={input_resolution})")
                pprint(e)
                print(traceback.format_exc())
                summary_rows.append({
                    "wm": width_multiplier,
                    "ir": input_resolution,
                    "name": name,
                    "cpb": cpb,
                    "max_features": 0,
                    "flops_sum": 0,
                })

# Write a single CSV with all models
if summary_rows:
    onnx_path = Settings.NAS_EVOSEARCH_SETTINGS['GEN_ONNX_FILE_PATH']
    os.makedirs(onnx_path, exist_ok=True)
    csv_path = os.path.join(onnx_path, Settings.NAS_SETTINGS_GENERAL['ARC']+"_spec_models_summary.csv")
    fieldnames = ["wm", "ir", "name", "cpb", "max_features", "flops_sum"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Convert cpb (list[list[int]]) to a compact string for the CSV
        for row in summary_rows:
            row_out = dict(row)
            if isinstance(row_out.get("cpb"), list):
                row_out["cpb"] = repr(row_out["cpb"])
            writer.writerow(row_out)
    print(f"[INFO] Wrote CSV summary: {csv_path}")
else:
    print("[INFO] No rows to write; CSV not created.")
# ======= END SPEC MODE =======
