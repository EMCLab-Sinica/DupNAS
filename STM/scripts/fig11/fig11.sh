set -euo pipefail

[[ "$OPTION" =~ ^(tflm|cubeai)-(f7|h7)$ ]] || { echo "Error: Invalid OPTION"; exit 1; }

ENGINE="${BASH_REMATCH[1]}"
BOARD="${BASH_REMATCH[2]}"

echo "Running engine: $ENGINE, board: $BOARD..."

bash "scripts/run_${ENGINE}_${BOARD}.sh" "scripts/fig11/${ENGINE}.txt" ram,latency

RESULTS_CSV="results/${ENGINE}_${BOARD}/results.csv"
awk -F, '
    NR == 1 {
        for (i = 1; i <= NF; i++) col[$i] = i
        next
    }
    $col["ram"] != "NA" { ram[++n_ram] = $col["ram"] }
    $col["latency"] != "NA" { latency[++n_latency] = $col["latency"] }
    function cmp(i1, v1, i2, v2) { return v1 - v2 }
    function percentile(a, n, p) {
        return a[int(1 + (n - 1) * p)]
    }
    function summary(name, a, n) {
        if (n == 0) return
        asort(a, a, "cmp")
        printf "%s: n=%d min=%.3f q1=%.3f median=%.3f q3=%.3f max=%.3f\n", \
            name, n, a[1], percentile(a, n, 0.25), percentile(a, n, 0.5), \
            percentile(a, n, 0.75), a[n]
    }
    END {
        print "=============================="
        print "Box plot values:"
        summary("ram", ram, n_ram)
        summary("latency", latency, n_latency)
    }
' "$RESULTS_CSV"
