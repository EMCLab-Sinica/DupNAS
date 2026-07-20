import csv
import sys
from pathlib import Path


def percentile(values, p):
    return values[int((len(values) - 1) * p)]


def summary(name, values):
    values = sorted(values)

    if not values:
        return f"{name}: no valid values"

    avg = sum(values) / len(values)

    return (
        f"{name:<25} "
        f"min={values[0]:>8.2f}  "
        f"q1={percentile(values, 0.25):>8.2f}  "
        f"median={percentile(values, 0.5):>8.2f}  "
        f"q3={percentile(values, 0.75):>8.2f}  "
        f"max={values[-1]:>8.2f}  "
        f"avg={avg:>8.2f}"
    )


def main(argv):
    if len(argv) != 2:
        raise SystemExit(f"Usage: {argv[0]} RESULTS_CSV")

    results_csv = Path(argv[1])
    log_path = Path("results/fig11_result.log")

    values = {
        "ram": [],
        "latency": [],
    }

    with results_csv.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            for field, field_values in values.items():
                value = row.get(field, "NA")

                if value not in ("NA", "", None):
                    field_values.append(float(value))

    lines = [
        "=" * 100,
        "Fig. 11 Results: Deployment Performance",
        "=" * 100,
        summary("Peak memory (KB)", values["ram"]),
        summary("Inference latency (s)", values["latency"]),
        "=" * 100,
    ]

    summary_text = "\n".join(lines)

    # Print to console
    print(summary_text)

    # Save the same summary to log
    log_path.write_text(summary_text + "\n", encoding="utf-8")

    print(f"[DONE] Saved Fig. 11 result log to: {log_path}")


if __name__ == "__main__":
    main(sys.argv)