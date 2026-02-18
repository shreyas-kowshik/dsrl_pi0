"""
Plot Eval_Step vs Success_Rate for each CSV in a wandb_gen_summary directory.

Usage:
    python tools/plot_eval_summaries.py wandb_gen_summary
    python tools/plot_eval_summaries.py /path/to/wandb_gen_summary
"""

import argparse
import os
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_csv(path):
    steps, rates = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["Eval_Step"]))
            rates.append(float(row["Success_Rate"]))
    return steps, rates


def plot_csv(csv_path, out_path):
    steps, rates = load_csv(csv_path)
    name = os.path.splitext(os.path.basename(csv_path))[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, rates, marker="o", linewidth=1.8, markersize=4, color="#2563eb")
    ax.set_xlabel("Eval Step", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(name.replace("_", " "), fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot eval summaries from CSV files.")
    parser.add_argument("summary_dir", help="Path to wandb_gen_summary directory")
    args = parser.parse_args()

    summary_dir = os.path.abspath(args.summary_dir)
    if not os.path.isdir(summary_dir):
        print(f"Error: directory not found: {summary_dir}", file=sys.stderr)
        sys.exit(1)

    plots_dir = os.path.join(summary_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    csv_files = sorted(
        f for f in os.listdir(summary_dir)
        if f.endswith(".csv") and os.path.isfile(os.path.join(summary_dir, f))
    )

    if not csv_files:
        print(f"No CSV files found in {summary_dir}")
        sys.exit(0)

    print(f"Found {len(csv_files)} CSV file(s). Plotting into {plots_dir}/")
    for fname in csv_files:
        csv_path = os.path.join(summary_dir, fname)
        out_path = os.path.join(plots_dir, os.path.splitext(fname)[0] + ".png")
        plot_csv(csv_path, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
