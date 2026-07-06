set -euo pipefail

[[ "$OPTION" =~ ^(neither|BPonly|PConly)-(shufflenet|mobilenet|inception)-vm(96|128|256)$ ]] || { echo "Error: Invalid OPTION"; exit 1; }

EXGRULE="${BASH_REMATCH[1]}"
MODEL="${BASH_REMATCH[2]}"
VM="${BASH_REMATCH[3]}"

echo "Executing exgrule: $EXGRULE, model: $MODEL, VM: $VM..."

cd DupNAS/HEtest
bash "run_${MODEL}_exgrule.sh" "$EXGRULE" "$VM"
python3.9 collect_allseed_report.py --exgrule "$EXGRULE" --model "$MODEL" --vm_setting "$VM"