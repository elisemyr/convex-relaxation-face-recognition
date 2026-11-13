"""
Runs every experiment and saves the results as CSV files in results/

Usage, from the project folder:
    python3 experiments/run_all.py                    # everything, on Yale B
    python3 experiments/run_all.py --quick            # small run to check the code
    python3 experiments/run_all.py --deep             # also run the FaceNet baseline
    python3 experiments/run_all.py --synthetic        # fake faces, no download needed

Experiments:
    E1  accuracy on clean faces vs feature size, for pixels, PCA and random projections
    E2  accuracy vs occlusion level, for each kind of occlusion
    E3  accuracy vs number of gallery images per person
    E4  effect of Robust PCA cleaning on the gallery
    E5  door decision: FAR and FRR curves, tau chosen on validation visitors
    E6  time per test face, measured during E5

The notebook only reads the CSV files, so it does not need to rerun anything
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config, quick_config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.evaluation import (accuracy, choose_tau, equal_error_rate, far_frr,  # noqa: E402
                            far_frr_curve, roc_auc)
from src.features import make_features  # noqa: E402
from src.occlusion import apply_occlusion  # noqa: E402
from src.pipeline import Pipeline, build_method  # noqa: E402
from src.src_classifier import CRCClassifier, SRCClassifier, access_decision  # noqa: E402
from src.baselines import EigenfacesSVM  # noqa: E402
from src.splits import split_from_config  # noqa: E402

BASE_METHODS = ["SRC", "SRC+occlusion", "CRC", "Eigenfaces+SVM", "Eigenfaces+NN"]


def log(msg):
    print(msg, flush=True)


def occluded(images, kind, level, seed):
    return images if level == 0 else apply_occlusion(images, kind, level, seed=seed)


def evaluate(method, images, labels, split, kind="none", level=0.0, seed=0):
    """Fit on the gallery and test on employees and test visitors. Returns a dict of metrics."""
    method.fit(images[split.gallery_idx], labels[split.gallery_idx])
    emp = method.run(occluded(images[split.emp_test_idx], kind, level, seed + 100))
    vis = method.run(occluded(images[split.vis_test_idx], kind, level, seed + 200))
    true = labels[split.emp_test_idx]
    _, far, frr = far_frr_curve(emp["pred"], true, emp["score"], vis["score"])
    return {
        "accuracy": accuracy(emp["pred"], true),
        "auc": roc_auc(far, frr),
        "eer": equal_error_rate(far, frr),
    }


# ----------------------------------------------------------------------------- E1
def exp_e1(cfg, images, labels):
    rows = []
    for seed in cfg.seeds:
        split = split_from_config(labels, cfg, seed)
        for kind in ["pixels", "pca", "random"]:
            for d in cfg.feature_dims:
                methods = {
                    "SRC": SRCClassifier(lam=cfg.lam_src, n_iter=cfg.fista_iter),
                    "CRC": CRCClassifier(lam=cfg.lam_crc),
                    "Eigenfaces+SVM": EigenfacesSVM(cfg.pca_components, seed=seed),
                }
                for name, clf in methods.items():
                    pipe = Pipeline(clf, make_features(kind, d, seed=seed))
                    res = evaluate(pipe, images, labels, split, seed=seed)
                    rows.append({"seed": seed, "features": kind, "d": d, "method": name, **res})
                    log(f"E1 seed={seed} {kind:7s} d={d:4d} {name:15s} acc={res['accuracy']:.3f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E2
def exp_e2(cfg, images, labels, methods):
    rows = []
    for seed in cfg.seeds:
        split = split_from_config(labels, cfg, seed)
        for kind in cfg.occlusion_kinds:
            for level in cfg.occlusion_levels:
                for name in methods:
                    res = evaluate(build_method(name, cfg, seed), images, labels, split, kind, level, seed)
                    rows.append({"seed": seed, "occlusion": kind, "level": level, "method": name, **res})
                    log(f"E2 seed={seed} {kind:7s} level={level:.1f} {name:20s} acc={res['accuracy']:.3f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E3
def exp_e3(cfg, images, labels, methods):
    rows = []
    conditions = [("none", 0.0), ("block", 0.3)]
    for seed in cfg.seeds:
        for k in cfg.gallery_sizes:
            split = split_from_config(labels, cfg, seed, gallery_size=k)
            for kind, level in conditions:
                for name in methods:
                    res = evaluate(build_method(name, cfg, seed), images, labels, split, kind, level, seed)
                    rows.append({"seed": seed, "gallery_size": k, "occlusion": kind, "level": level,
                                 "method": name, **res})
                    log(f"E3 seed={seed} k={k:2d} {kind:5s} {name:20s} acc={res['accuracy']:.3f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E4
def exp_e4(cfg, images, labels):
    rows = []
    pairs = [("SRC", "SRC+RPCA"), ("SRC+occlusion", "SRC+occlusion+RPCA"), ("CRC", "CRC+RPCA")]
    conditions = [("none", 0.0), ("block", 0.3), ("mask", 0.4)]
    for seed in cfg.seeds:
        split = split_from_config(labels, cfg, seed)
        for kind, level in conditions:
            for base, cleaned in pairs:
                for name, rpca_on in [(base, False), (cleaned, True)]:
                    res = evaluate(build_method(name, cfg, seed), images, labels, split, kind, level, seed)
                    rows.append({"seed": seed, "occlusion": kind, "level": level, "method": base,
                                 "rpca": rpca_on, **res})
                    log(f"E4 seed={seed} {kind:5s} {name:20s} acc={res['accuracy']:.3f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E5 and E6
def exp_e5(cfg, images, labels, methods):
    """
    For each method: choose tau on the validation visitors so that FAR <= target,
    then report FAR and FRR on the test employees and test visitors.
    Also saves the full curves for the ROC figure, and the time per face for E6.
    """
    summary, curves = [], []
    conditions = [("none", 0.0), ("mask", 0.4), ("glasses", 0.2)]
    for seed in cfg.seeds:
        split = split_from_config(labels, cfg, seed)
        for kind, level in conditions:
            for name in methods:
                method = build_method(name, cfg, seed).fit(images[split.gallery_idx], labels[split.gallery_idx])
                run = lambda idx, s: method.run(occluded(images[idx], kind, level, seed + s))
                emp_val, vis_val = run(split.emp_val_idx, 300), run(split.vis_val_idx, 400)
                emp_test, vis_test = run(split.emp_test_idx, 100), run(split.vis_test_idx, 200)
                true_test = labels[split.emp_test_idx]

                tau = choose_tau(vis_val["score"], cfg.target_far)
                far, frr = far_frr(tau, emp_test["pred"], true_test, emp_test["score"], vis_test["score"])

                # Three-way decision: accept, reject or badge.
                dec_emp = access_decision(emp_test["pred"], emp_test["score"], tau,
                                          emp_test["occluded_fraction"], cfg.rho_fallback)
                dec_vis = access_decision(vis_test["pred"], vis_test["score"], tau,
                                          vis_test["occluded_fraction"], cfg.rho_fallback)

                taus, far_c, frr_c = far_frr_curve(emp_test["pred"], true_test,
                                                   emp_test["score"], vis_test["score"])
                summary.append({
                    "seed": seed, "occlusion": kind, "level": level, "method": name,
                    "tau": tau, "far_test": far, "frr_test": frr,
                    "accuracy": accuracy(emp_test["pred"], true_test),
                    "auc": roc_auc(far_c, frr_c), "eer": equal_error_rate(far_c, frr_c),
                    "employees_badge_share": float(np.mean(dec_emp == "badge")),
                    "visitors_badge_share": float(np.mean(dec_vis == "badge")),
                    "ms_per_face": 1000 * (emp_test["time_per_face"] + vis_test["time_per_face"]) / 2,
                })
                curves.append(pd.DataFrame({"seed": seed, "occlusion": kind, "level": level,
                                            "method": name, "tau": taus, "far": far_c, "frr": frr_c}))
                log(f"E5 seed={seed} {kind:7s} {name:20s} FAR={far:.3f} FRR={frr:.3f}")
    return pd.DataFrame(summary), pd.concat(curves, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small run to check the code")
    parser.add_argument("--synthetic", action="store_true", help="use fake faces instead of Yale B")
    parser.add_argument("--deep", action="store_true", help="also run the FaceNet baseline")
    parser.add_argument("--exp", nargs="+", default=["E1", "E2", "E3", "E4", "E5"])
    args = parser.parse_args()

    cfg = quick_config() if args.quick else Config()
    out_dir = cfg.results_dir / ("synthetic" if args.synthetic else "yaleb")
    out_dir.mkdir(parents=True, exist_ok=True)

    images, labels, _ = load_dataset(cfg, synthetic=args.synthetic)
    log(f"Loaded {len(images)} images of {len(np.unique(labels))} people, size {images.shape[1:]}")

    methods = BASE_METHODS + (["FaceNet+NN"] if args.deep else [])

    if "E1" in args.exp:
        exp_e1(cfg, images, labels).to_csv(out_dir / "e1_feature_dims.csv", index=False)
    if "E2" in args.exp:
        exp_e2(cfg, images, labels, methods).to_csv(out_dir / "e2_occlusion.csv", index=False)
    if "E3" in args.exp:
        exp_e3(cfg, images, labels, methods).to_csv(out_dir / "e3_gallery_size.csv", index=False)
    if "E4" in args.exp:
        exp_e4(cfg, images, labels).to_csv(out_dir / "e4_rpca.csv", index=False)
    if "E5" in args.exp:
        summary, curves = exp_e5(cfg, images, labels, methods)
        summary.to_csv(out_dir / "e5_access_summary.csv", index=False)
        curves.to_csv(out_dir / "e5_access_curves.csv", index=False)
    log(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()