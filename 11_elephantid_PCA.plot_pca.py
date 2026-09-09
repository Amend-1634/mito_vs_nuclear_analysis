"""
Plot PC1 vs PC2 from EMU output, with axis labels showing % variance
explained (relative to all estimated PCs) and full italic species names
in the legend.

Usage:
    python plot_pca.py <prefix>

Expects <prefix>.eigvecs and <prefix>.eigvals in the same directory
(EMU's default output naming, e.g. all_mammoths_pca.eigvecs / .eigvals).
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd

# ---- map FID prefix (text before first "-") to full species name ----
# Edit this dict if your sample naming or taxa differ.
FULL_NAMES = {
    "Mamericanum": "Mammut americanum",
    "Mcolumbi": "Mammuthus columbi",
    "Pantiquus": "Palaeoloxodon antiquus",
    "Mammuthus": "Mammuthus primigenius",
    "Unknown": "Unknown",
}


def load_eigvecs(path):
    df = pd.read_csv(path, sep=r"\s+", comment=None)
    df = df.rename(columns={df.columns[0]: "FID"})
    df["species"] = df["FID"].apply(
        lambda x: x.split("-")[0] if "-" in x else "Unknown"
    )
    return df


def load_eigvals(path):
    with open(path) as f:
        vals = [float(line.strip()) for line in f if line.strip()]
    return vals


def plot_pca(eigvecs_path, eigvals_path, out_path):
    df = load_eigvecs(eigvecs_path)
    eigvals = load_eigvals(eigvals_path)
    total_var = sum(eigvals)
    pc1_pct = eigvals[0] / total_var * 100
    pc2_pct = eigvals[1] / total_var * 100

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = list(plt.cm.tab10.colors)
    species_order = sorted(df["species"].unique())

    # swap colors for Pantiquus (Palaeoloxodon antiquus) and Unknown
    color_map = {sp: colors[i % len(colors)] for i, sp in enumerate(species_order)}
    if "Pantiquus" in color_map and "Unknown" in color_map:
        color_map["Pantiquus"], color_map["Unknown"] = color_map["Unknown"], color_map["Pantiquus"]

    for sp in species_order:
        sub = df[df["species"] == sp]
        name = FULL_NAMES.get(sp, sp)
        if name != "Unknown":
            italic_name = r"$\mathit{" + name.replace(" ", r"\ ") + r"}$"
            label = f"{italic_name} (n={len(sub)})"
        else:
            label = f"{name} (n={len(sub)})"
        alpha = 1.0 if sp == "Unknown" else 0.85
        ax.scatter(
            sub["PC1"], sub["PC2"],
            label=label,
            color=color_map[sp],
            s=117, edgecolor="k", linewidth=0.5, alpha=alpha,
        )

    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}%)", fontsize=16)
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)

    leg = ax.legend(loc="best", fontsize=13)

    plt.tight_layout()
    base = out_path.rsplit(".", 1)[0]
    for ext in ("png", "jpg", "pdf"):
        plt.savefig(f"{base}.{ext}", dpi=300)
    print(f"Saved: {base}.png, {base}.jpg, {base}.pdf")
    print(f"PC1: {pc1_pct:.2f}%  PC2: {pc2_pct:.2f}%  "
          f"(of total variance across {len(eigvals)} estimated PCs, sum={total_var:.2f})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_pca.py <prefix>")
        print("  expects <prefix>.eigvecs and <prefix>.eigvals")
        sys.exit(1)
    prefix = sys.argv[1]
    plot_pca(f"{prefix}.eigvecs", f"{prefix}.eigvals", "pca_plot.png")