"""Plot FoM vs simulation budget from NPE results on Modal volume.

Usage:
    .venv/bin/modal run scripts/plot_fom_budget.py
    # or if results are already downloaded:
    python scripts/plot_fom_budget.py --from-json results.json
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_fom(budgets, fom_means, fom_stds, output):
    order = np.argsort(budgets)
    budgets = budgets[order]
    fom_means = fom_means[order]
    fom_stds = fom_stds[order]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        budgets,
        fom_means,
        yerr=fom_stds,
        fmt="o-",
        color="C0",
        markersize=8,
        capsize=4,
        linewidth=1.5,
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of compressor training simulations", fontsize=14)
    ax.set_ylabel("Figure of Merit (FoM)", fontsize=14)
    ax.set_title("Inference quality vs simulation budget", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Plot saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot FoM vs budget from NPE results")
    parser.add_argument("--from-json", default=None, help="Path to local JSON file with results")
    parser.add_argument("--output", default="fom_budget_scan.pdf", help="Output plot file")
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json) as f:
            all_results = json.load(f)
    else:
        # Fetch from Modal volume
        import modal

        app = modal.App.lookup("cosmoford-npe-budget-scan")
        load_fn = app.registered_functions["load_all_results"]
        all_results = load_fn.remote()

    if not all_results:
        print("No results found.")
        return

    budgets = np.array([r["budget"] for r in all_results])
    fom_means = np.array([r["fom_mean"] for r in all_results])
    fom_stds = np.array([r["fom_std"] for r in all_results])

    for r in sorted(all_results, key=lambda x: x["budget"]):
        print(f"  budget={r['budget']:>6d}: FoM = {r['fom_mean']:.2f} ± {r['fom_std']:.2f} "
              f"(val_nll={r['best_val_nll']:.4f})")

    plot_fom(budgets, fom_means, fom_stds, args.output)


if __name__ == "__main__":
    main()
