set -euo pipefail

[[ "$OPTION" =~ ^(shufflenet|mobilenet|inception)-vm(96|128|256)$ ]] || { echo "Error: Invalid OPTION"; exit 1; }

MODEL="${BASH_REMATCH[1]}"
VM="${BASH_REMATCH[2]}"

echo "Running model: $MODEL, VM: $VM..."

cd "DupNAS/sample_onnx/$MODEL"
bash "run_allsamples_vm${VM}.sh" ../outputs python3.9

cd ..
python3.9 analyze_all_output_dupnasa.py --output_dir ./outputs --model "$MODEL" --vm "$VM"
