"""
Makes every figure of the report.

Usage, from the project folder:
    python3 experiments/plots.py               # figures from the Yale B results
    python3 experiments/plots.py --synthetic   # figures from the synthetic results

The result figures only read the CSV files written by run_all.py.
The example figures load a few faces to show occlusions, Robust PCA and
one SRC decision. Everything is saved in results/<dataset>/figures/.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.features import PixelFeatures  # noqa: E402
from src.occlusion import apply_occlusion  # noqa: E402
from src.rpca import rpca  # noqa: E402
from src.splits import split_from_config  # noqa: E402
from src.src_classifier import SRCClassifier  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


# One fixed color per method, so a method looks the same in every figure.
COLORS = {"SRC": "#d62728", "SRC+occlusion": "#9467bd", "SRC+occlusion+RPCA": "#8c564b",
          "CRC": "#1f77b4", "Eigenfaces+SVM": "#2ca02c", "Eigenfaces+NN": "#ff7f0e",
          "FaceNet+NN": "#555555"}


def mean_std(df, by, value="accuracy"):
    return df.groupby(by)[value].agg(["mean", "std"]).reset_index().fillna(0.0)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ----------------------------------------------------------------------------- result figures
def plot_e1(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, kind in zip(axes, ["pixels", "pca", "random"]):
        stats = mean_std(df[df.features == kind], ["method", "d"])
        for method, g in stats.groupby("method"):
            ax.errorbar(g.d, g["mean"], yerr=g["std"], marker="o", capsize=3, label=method, color=COLORS.get(method))
        ax.set_xscale("log")
        ax.set_title({"pixels": "Downsampled pixels", "pca": "PCA", "random": "Random projection"}[kind])
        ax.set_xlabel("feature dimension d")
    axes[0].set_ylabel("identification accuracy")
    axes[0].legend()
    save(fig, out / "e1_feature_dims.png")


def plot_e2(df, out):
    kinds = list(df.occlusion.unique())
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.4 * len(kinds), 3.6), sharey=True, squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        stats = mean_std(df[df.occlusion == kind], ["method", "level"])
        for method, g in stats.groupby("method"):
            ax.errorbar(100 * g.level, g["mean"], yerr=g["std"], marker="o", capsize=3, label=method, color=COLORS.get(method))
        ax.set_title(f"occlusion: {kind}")
        ax.set_xlabel("% of the face hidden")
    axes[0][0].set_ylabel("identification accuracy")
    axes[0][0].legend(fontsize=8)
    save(fig, out / "e2_occlusion.png")


def plot_e3(df, out):
    conds = df[["occlusion", "level"]].drop_duplicates().values
    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 3.6), sharey=True, squeeze=False)
    for ax, (kind, level) in zip(axes[0], conds):
        sub = df[(df.occlusion == kind) & (df.level == level)]
        stats = mean_std(sub, ["method", "gallery_size"])
        for method, g in stats.groupby("method"):
            ax.errorbar(g.gallery_size, g["mean"], yerr=g["std"], marker="o", capsize=3, label=method, color=COLORS.get(method))
        ax.set_title("clean faces" if level == 0 else f"{kind}, {int(100 * level)}% hidden")
        ax.set_xlabel("gallery images per person")
    axes[0][0].set_ylabel("identification accuracy")
    axes[0][0].legend(fontsize=8)
    save(fig, out / "e3_gallery_size.png")


def plot_e4(df, out):
    df = df.assign(condition=np.where(df.level == 0, "clean",
                                      df.occlusion + " " + (100 * df.level).astype(int).astype(str) + "%"))
    stats = mean_std(df, ["condition", "method", "rpca"])
    conditions = list(dict.fromkeys(df.condition))
    methods = list(dict.fromkeys(df.method))
    fig, axes = plt.subplots(1, len(conditions), figsize=(4 * len(conditions), 3.4), sharey=True, squeeze=False)
    width = 0.38
    for ax, cond in zip(axes[0], conditions):
        x = np.arange(len(methods))
        for j, (rpca_on, label) in enumerate([(False, "raw gallery"), (True, "RPCA-cleaned gallery")]):
            s = stats[(stats.condition == cond) & (stats.rpca == rpca_on)].set_index("method").loc[methods]
            ax.bar(x + (j - 0.5) * width, s["mean"], width, yerr=s["std"], capsize=3, label=label)
        ax.set_xticks(x, methods, rotation=20)
        ax.set_title(cond)
    axes[0][0].set_ylabel("identification accuracy")
    axes[0][0].legend(fontsize=8)
    save(fig, out / "e4_rpca.png")


def plot_e5(summary, curves, out):
    conds = summary[["occlusion", "level"]].drop_duplicates().values
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.8), sharey=True, squeeze=False)
    for ax, (kind, level) in zip(axes[0], conds):
        sub = curves[(curves.occlusion == kind) & (curves.level == level) & (curves.seed == curves.seed.min())]
        for method, g in sub.groupby("method"):
            g = g.sort_values("far")
            ax.plot(g.far, 1 - g.frr, label=method, color=COLORS.get(method))
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(0, 1)
        ax.set_title("clean faces" if level == 0 else f"{kind}, {int(100 * level)}% hidden")
        ax.set_xlabel("false accept rate")
    axes[0][0].set_ylabel("1 - false reject rate")
    axes[0][0].legend(fontsize=8)
    save(fig, out / "e5_roc.png")

    table = (summary.groupby(["occlusion", "level", "method"])
             [["far_test", "frr_test", "auc", "eer", "employees_badge_share", "ms_per_face"]]
             .mean().round(3))
    table.to_csv(out / "e5_table.csv")
    print(table)


def plot_e6(summary, out):
    t = summary.groupby("method")["ms_per_face"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(t.index, t.values)
    ax.set_xlabel("milliseconds per test face")
    ax.set_title("Time for one decision at the door")
    save(fig, out / "e6_time.png")


# ----------------------------------------------------------------------------- example figures
def plot_examples(cfg, synthetic, out):
    images, labels, _ = load_dataset(cfg, synthetic=synthetic)
    split = split_from_config(labels, cfg, seed=0)
    face = images[split.emp_test_idx[:1]]

    # 1. One face with each kind of occlusion.
    settings = [("none", 0.0), ("block", 0.3), ("mask", 0.45), ("glasses", 0.2), ("pixels", 0.3)]
    fig, axes = plt.subplots(1, len(settings), figsize=(2.2 * len(settings), 2.6))
    for ax, (kind, level) in zip(axes, settings):
        img = face[0] if level == 0 else apply_occlusion(face, kind, level, seed=0)[0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(kind if level == 0 else f"{kind} {int(100 * level)}%")
        ax.axis("off")
    save(fig, out / "example_occlusions.png")

    # 2. Robust PCA on the gallery of one person: M = L + S.
    person = split.employees[0]
    idx = split.gallery_idx[labels[split.gallery_idx] == person]
    feats = PixelFeatures(cfg.work_size)
    M = feats.transform(images[idx]).T
    L, S, info = rpca(M)
    h, w = cfg.work_size
    show = min(5, M.shape[1])
    fig, axes = plt.subplots(3, show, figsize=(1.8 * show, 6))
    for j in range(show):
        for i, (mat, name) in enumerate([(M, "original M"), (L, "low rank L"), (S, "sparse S")]):
            axes[i, j].imshow(mat[:, j].reshape(h, w), cmap="gray")
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(name, loc="left", fontsize=9)
    fig.suptitle(f"Robust PCA on one person, rank of L = {info['rank']}")
    save(fig, out / "example_rpca.png")

    # 3. One SRC decision with the occlusion model.
    clf = SRCClassifier(lam=cfg.lam_src, occlusion=True, n_iter=cfg.fista_iter,
                        e_threshold=cfg.e_pixel_threshold)
    clf.fit(feats.transform(images[split.gallery_idx]), labels[split.gallery_idx])
    test_img = apply_occlusion(face, "block", 0.3, seed=0)
    res = clf.explain(feats.transform(test_img))
    coef, e = res["coef"][:, 0], res["e"][:, 0]
    pred = res["pred"][0]
    rebuilt = clf.Phi_[:, clf.labels_ == pred] @ coef[clf.labels_ == pred]

    fig = plt.figure(figsize=(12, 3))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 3, 1, 1, 1])
    ax = fig.add_subplot(grid[0])
    ax.imshow(feats.transform(test_img)[0].reshape(h, w), cmap="gray")
    ax.set_title("test face")
    ax.axis("off")
    ax = fig.add_subplot(grid[1])
    ax.bar(np.arange(len(coef)), coef, width=1.0)
    ax.set_title(f"coefficients x, SCI = {res['score'][0]:.2f}, true = {labels[split.emp_test_idx[0]]}, "
                 f"predicted = {pred}")
    ax.set_xlabel("gallery image, sorted by person")
    for ax_i, (vec, name) in zip(range(2, 5), [(rebuilt, "rebuilt face"), (e, "error e"),
                                               (np.abs(e) > cfg.e_pixel_threshold / np.sqrt(len(e)),
                                                f"occluded: {100 * res['occluded_fraction'][0]:.0f}%")]):
        ax = fig.add_subplot(grid[ax_i])
        ax.imshow(np.asarray(vec, dtype=float).reshape(h, w), cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    save(fig, out / "example_decision.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--no-examples", action="store_true", help="skip the figures that load images")
    args = parser.parse_args()

    cfg = Config()
    res_dir = cfg.results_dir / ("synthetic" if args.synthetic else "yaleb")
    out = res_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "e1": res_dir / "e1_feature_dims.csv", "e2": res_dir / "e2_occlusion.csv",
        "e3": res_dir / "e3_gallery_size.csv", "e4": res_dir / "e4_rpca.csv",
        "e5s": res_dir / "e5_access_summary.csv", "e5c": res_dir / "e5_access_curves.csv",
    }
    if files["e1"].exists():
        plot_e1(pd.read_csv(files["e1"]), out)
    if files["e2"].exists():
        plot_e2(pd.read_csv(files["e2"]), out)
    if files["e3"].exists():
        plot_e3(pd.read_csv(files["e3"]), out)
    if files["e4"].exists():
        plot_e4(pd.read_csv(files["e4"]), out)
    if files["e5s"].exists() and files["e5c"].exists():
        summary = pd.read_csv(files["e5s"])
        plot_e5(summary, pd.read_csv(files["e5c"]), out)
        plot_e6(summary, out)
    if not args.no_examples:
        plot_examples(cfg, args.synthetic, out)


if __name__ == "__main__":
    main()