import csv
import sys


def percentile(values, p):
    return values[int((len(values) - 1) * p)]


def summary(name, values):
    values = sorted(values)
    print(
        f"{name}: n={len(values)} min={values[0]:.3f} "
        f"q1={percentile(values, 0.25):.3f} "
        f"median={percentile(values, 0.5):.3f} "
        f"q3={percentile(values, 0.75):.3f} max={values[-1]:.3f}"
    )


def main(argv):
    if len(argv) != 2:
        raise SystemExit(f"Usage: {argv[0]} RESULTS_CSV")

    values = {"ram": [], "latency": []}
    with open(argv[1], newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            for field, field_values in values.items():
                value = row.get(field, "NA")
                if value != "NA":
                    field_values.append(float(value))

    print("==============================")
    print("Box plot values:")
    summary("ram", values["ram"])
    summary("latency", values["latency"])


if __name__ == "__main__":
    main(sys.argv)
