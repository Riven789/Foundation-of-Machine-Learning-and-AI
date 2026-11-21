"""
Homework 3, Problem 3: generalization of gradient boosting as the number of trees grows.

For dset0 and dset1 (first column = binary label, other 5 columns = covariates):
  (a)/(b) stratified 70/30 train/test split (seed 0). Fit sklearn GradientBoostingClassifier
          with 1, 10, 100, 300, 1000 trees (default learning_rate=0.1, max_depth=3) and report
          train/test accuracy and log-loss. The same thing is repeated over 30 random splits.
          staged_predict gives the whole curve for 1..1000 trees, and it is checked to match
          the separately fit models exactly. Stumps (max_depth=1) are run as a supplement.
  (c)     diagnostics for the difference between the datasets:
          * 5-fold CV accuracy of a linear model (logistic regression) vs flexible models.
          * learning curves (accuracy vs training-set size) of the same models.
          * paired (same split) differences in test accuracy between numbers of trees.
          * controlled label-noise experiments:
              - dset0 with 20% of the TRAINING labels flipped (test labels left clean);
              - dset1 "denoised": labels replaced by the logistic-regression prediction,
                which gives the same nearly linear boundary with no label noise.

Usage:
    pip install numpy pandas scikit-learn matplotlib
    python problem3.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit, cross_val_score,
                                     learning_curve, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DSETS = ["dset0", "dset1"]
N_TREES = [1, 10, 100, 300, 1000]
N_MAX = max(N_TREES)
TEST_SIZE = 0.3
N_REPEAT = 30
SEED = 0

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 10,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(name):
    d = pd.read_csv(DATA / f"{name}.csv", header=None).to_numpy(float)
    return d[:, 1:], d[:, 0].astype(int)


def split(X, y, seed):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)


def staged_curves(X_tr, y_tr, X_te, y_te, seed, max_depth=3, y_te_eval=None):
    """Fit one N_MAX-tree model; return train/test accuracy and log-loss after each stage."""
    y_te_eval = y_te if y_te_eval is None else y_te_eval
    m = GradientBoostingClassifier(n_estimators=N_MAX, max_depth=max_depth, random_state=seed)
    m.fit(X_tr, y_tr)
    out = {}
    for tag, X, y in [("train", X_tr, y_tr), ("test", X_te, y_te_eval)]:
        P = np.array(list(m.staged_predict_proba(X)))[:, :, 1]  # (N_MAX, n)
        P = np.clip(P, 1e-15, 1 - 1e-15)
        out[f"{tag}_acc"] = ((P > 0.5) == y).mean(axis=1)
        out[f"{tag}_logloss"] = -(y * np.log(P) + (1 - y) * np.log(1 - P)).mean(axis=1)
    return out


def repeated(X, y, max_depth=3, flip=0.0, n_rep=N_REPEAT):
    """Mean and standard error over n_rep random splits of the staged curves.
    flip > 0 flips that fraction of the training labels (test labels stay clean)."""
    runs = []
    for s in range(n_rep):
        X_tr, X_te, y_tr, y_te = split(X, y, s)
        if flip > 0:
            rng = np.random.default_rng(1000 + s)
            f = rng.random(len(y_tr)) < flip
            y_tr = np.where(f, 1 - y_tr, y_tr)
        runs.append(staged_curves(X_tr, y_tr, X_te, y_te, s, max_depth=max_depth))
    keys = runs[0].keys()
    mean = {k: np.mean([r[k] for r in runs], axis=0) for k in keys}
    se = {k: np.std([r[k] for r in runs], axis=0, ddof=1) / np.sqrt(n_rep) for k in keys}
    mean["runs_test_acc"] = np.array([r["test_acc"] for r in runs])  # (n_rep, N_MAX)
    return mean, se


def table_at(mean, se, label):
    rows = []
    for n in N_TREES:
        rows.append({"setting": label, "n_trees": n,
                     **{k: mean[k][n - 1] for k in mean if k != "runs_test_acc"},
                     **{f"{k}_se": se[k][n - 1] for k in ("test_acc", "test_logloss")}})
    best = int(np.argmax(mean["test_acc"])) + 1
    return pd.DataFrame(rows), best


def main():
    single_rows, rep_tables, curves, stump_tables, diag_rows = [], [], {}, [], []
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    for name in DSETS:
        X, y = load(name)
        print(f"\n===== {name}: n={len(y)}, p={X.shape[1]}, P(y=1)={y.mean():.3f} =====")

        # ---- (a)/(b): the five requested models on one split ----
        X_tr, X_te, y_tr, y_te = split(X, y, SEED)
        staged = staged_curves(X_tr, y_tr, X_te, y_te, SEED)
        for n in N_TREES:
            m = GradientBoostingClassifier(n_estimators=n, random_state=SEED).fit(X_tr, y_tr)
            p_tr, p_te = m.predict_proba(X_tr)[:, 1], m.predict_proba(X_te)[:, 1]
            row = {"dataset": name, "n_trees": n,
                   "train_acc": np.mean((p_tr > 0.5) == y_tr), "test_acc": np.mean((p_te > 0.5) == y_te),
                   "train_logloss": log_loss(y_tr, p_tr, labels=[0, 1]),
                   "test_logloss": log_loss(y_te, p_te, labels=[0, 1])}
            assert abs(row["test_acc"] - staged["test_acc"][n - 1]) < 1e-12  # staged == refit
            single_rows.append(row)

        # ---- repeated splits, default trees (depth 3) and stumps (depth 1) ----
        mean, se = repeated(X, y)
        curves[name] = (mean, se)
        t, best = table_at(mean, se, f"{name} (depth 3)")
        t["best_n_by_mean_test_acc"] = best
        rep_tables.append(t)
        print(t.round(3).to_string(index=False))

        m1, s1 = repeated(X, y, max_depth=1)
        curves[name + "_stumps"] = (m1, s1)
        t1, best1 = table_at(m1, s1, f"{name} (stumps)")
        t1["best_n_by_mean_test_acc"] = best1
        stump_tables.append(t1)

        # ---- (c) diagnostics: how much of the signal is linear? ----
        models = {
            "logistic regression (linear)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            "RBF-SVM (C=10)": make_pipeline(StandardScaler(), SVC(C=10)),
            "random forest (500 trees)": RandomForestClassifier(500, random_state=SEED),
            "GB, 100 trees": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        }
        for mname, model in models.items():
            diag_rows.append({"dataset": name, "model": mname,
                              "cv_accuracy": cross_val_score(model, X, y, cv=cv).mean()})
        diag_rows.append({"dataset": name, "model": "majority class", "cv_accuracy": max(y.mean(), 1 - y.mean())})

    single = pd.DataFrame(single_rows)
    single.to_csv(HERE / "problem3_single_split.csv", index=False)
    pd.concat(rep_tables).to_csv(HERE / "problem3_repeated_splits.csv", index=False)
    pd.concat(stump_tables).to_csv(HERE / "problem3_repeated_splits_stumps.csv", index=False)
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(HERE / "problem3_cv_diagnostics.csv", index=False)
    print("\nsingle split (seed 0):\n", single.round(3).to_string(index=False))
    print("\nstumps:\n", pd.concat(stump_tables).round(3).to_string(index=False))
    print("\nCV diagnostics:\n", diag.pivot(index="model", columns="dataset", values="cv_accuracy").round(3))

    # ---- paired differences in test accuracy (same split, different number of trees) ----
    paired = []
    for key in DSETS + [d + "_stumps" for d in DSETS]:
        A = curves[key][0]["runs_test_acc"]
        for a, b in [(10, 100), (100, 300), (100, 1000), (300, 1000)]:
            diff = A[:, b - 1] - A[:, a - 1]
            paired.append({"setting": key, "from": a, "to": b, "mean_diff": diff.mean(),
                           "se_diff": diff.std(ddof=1) / np.sqrt(len(diff)),
                           "frac_splits_worse": np.mean(diff < 0)})
    paired = pd.DataFrame(paired)
    paired.to_csv(HERE / "problem3_paired_differences.csv", index=False)
    print("\npaired test-accuracy differences acc(to) - acc(from), over splits:\n",
          paired.round(4).to_string(index=False))

    # ---- learning curves: does more data help (signal-limited) or not (noise-limited)? ----
    lc_models = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "RBF-SVM": make_pipeline(StandardScaler(), SVC(C=10)),
        "random forest": RandomForestClassifier(300, random_state=SEED),
        "GB, 100 trees": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    }
    lc_sizes = [50, 100, 200, 300, 400]
    lc_rows = []
    for name in DSETS:
        X, y = load(name)
        for mname, model in lc_models.items():
            sz, _, te = learning_curve(model, X, y, train_sizes=lc_sizes, n_jobs=-1,
                                       cv=StratifiedShuffleSplit(20, test_size=0.2, random_state=SEED))
            for n_, a in zip(sz, te.mean(axis=1)):
                lc_rows.append({"dataset": name, "model": mname, "n_train": int(n_), "test_acc": a})
    lc = pd.DataFrame(lc_rows)
    lc.to_csv(HERE / "problem3_learning_curves.csv", index=False)
    print("\nlearning curves (mean test accuracy over 20 splits):\n",
          lc.pivot_table(index=["dataset", "model"], columns="n_train", values="test_acc").round(3))

    # ---- (c) controlled label-noise experiments ----
    X0, y0 = load("dset0")
    X1, y1 = load("dset1")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X1, y1)
    y1_clean = lr.predict(X1)
    print(f"\ndset1: logistic prediction agrees with the given label on {np.mean(y1_clean == y1):.3f} of points")
    noise_settings = {
        "dset0 (original)": (X0, y0, 0.0),
        "dset0, 20% train labels flipped": (X0, y0, 0.2),
        "dset1 (original)": (X1, y1, 0.0),
        "dset1, denoised (logistic labels)": (X1, y1_clean, 0.0),
    }
    noise_tables = []
    for label, (X, y, flip) in noise_settings.items():
        if label.endswith("(original)"):
            mean, se = curves[label.split()[0]]
        else:
            mean, se = repeated(X, y, flip=flip)
        curves[label] = (mean, se)
        t, best = table_at(mean, se, label)
        t["best_n_by_mean_test_acc"] = best
        noise_tables.append(t)
    noise = pd.concat(noise_tables)
    noise.to_csv(HERE / "problem3_label_noise_experiments.csv", index=False)
    print("\nlabel-noise experiments:\n",
          noise[["setting", "n_trees", "train_acc", "test_acc", "test_acc_se", "best_n_by_mean_test_acc"]]
          .round(3).to_string(index=False))

    # ---------------- figures ----------------
    n_axis = np.arange(1, N_MAX + 1)

    def acc_panel(ax, mean, se, title):
        ax.plot(n_axis, mean["train_acc"], color=AQUA, label="train")
        ax.plot(n_axis, mean["test_acc"], color=BLUE, label="test")
        ax.fill_between(n_axis, mean["test_acc"] - 2 * se["test_acc"], mean["test_acc"] + 2 * se["test_acc"],
                        color=BLUE, alpha=0.15, lw=0)
        b = int(np.argmax(mean["test_acc"]))
        ax.scatter([n_axis[b]], [mean["test_acc"][b]], color=INK, s=22, zorder=5)
        ax.annotate(f"max test acc {mean['test_acc'][b]:.3f}\nat {n_axis[b]} trees", (n_axis[b], mean["test_acc"][b]),
                    xytext=(0, -30), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        for n in N_TREES:
            ax.axvline(n, color=GRID, lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("number of trees (log scale)")

    # accuracy and log-loss, depth 3, both datasets
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0))
    for j, name in enumerate(DSETS):
        mean, se = curves[name]
        acc_panel(axes[0, j], mean, se, f"{name}: accuracy")
        axes[0, j].set_ylim(0.5, 1.02)
        axes[0, j].set_ylabel("accuracy")
        ax = axes[1, j]
        ax.plot(n_axis, mean["train_logloss"], color=AQUA, label="train")
        ax.plot(n_axis, mean["test_logloss"], color=BLUE, label="test")
        ax.fill_between(n_axis, mean["test_logloss"] - 2 * se["test_logloss"],
                        mean["test_logloss"] + 2 * se["test_logloss"], color=BLUE, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.2)
        ax.set_title(f"{name}: log-loss (deviance)")
        ax.set_xlabel("number of trees (log scale)")
        ax.set_ylabel("log-loss")
    for ax in axes.flat:
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Gradient boosting (depth-3 trees, learning rate 0.1), mean over {N_REPEAT} random 70/30 splits "
                 f"(band = ±2 s.e.)", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(HERE / "problem3_curves.png")
    plt.close(fig)

    # test accuracy only, zoomed, both datasets, depth 3 and stumps
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, name in zip(axes, DSETS):
        for key, color, lab in [(name, BLUE, "depth-3 trees (default)"), (name + "_stumps", ORANGE, "stumps (depth 1)")]:
            mean, se = curves[key]
            ax.plot(n_axis, mean["test_acc"], color=color, label=lab)
            ax.fill_between(n_axis, mean["test_acc"] - 2 * se["test_acc"], mean["test_acc"] + 2 * se["test_acc"],
                            color=color, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xlim(5, N_MAX)
        ax.set_ylim(0.6, 0.86)
        ax.set_title(f"{name}: test accuracy")
        ax.set_xlabel("number of trees (log scale)")
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0].set_ylabel("test accuracy")
    fig.tight_layout()
    fig.savefig(HERE / "problem3_test_acc_zoom.png")
    plt.close(fig)

    # label-noise experiments
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    pairs = [("dset0 (original)", "dset0, 20% train labels flipped"),
             ("dset1 (original)", "dset1, denoised (logistic labels)")]
    for ax, (a, b) in zip(axes, pairs):
        for key, color in [(a, BLUE), (b, ORANGE)]:
            mean, se = curves[key]
            ax.plot(n_axis, mean["test_acc"], color=color, label=key)
            ax.fill_between(n_axis, mean["test_acc"] - 2 * se["test_acc"], mean["test_acc"] + 2 * se["test_acc"],
                            color=color, alpha=0.15, lw=0)
            k = int(np.argmax(mean["test_acc"]))
            ax.scatter([n_axis[k]], [mean["test_acc"][k]], color=color, s=24, zorder=5,
                       edgecolors="white", linewidths=0.8)
        ax.set_xscale("log")
        ax.set_xlim(5, N_MAX)
        ax.set_xlabel("number of trees (log scale)")
        ax.set_ylabel("test accuracy (clean test labels)" if ax is axes[0] else "test accuracy")
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0].set_title("Adding label noise to dset0")
    axes[1].set_title("Removing label noise from dset1")
    fig.tight_layout()
    fig.savefig(HERE / "problem3_label_noise.png")
    plt.close(fig)

    # learning curves
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    colors = dict(zip(lc_models, [BLUE, ORANGE, AQUA, "#eda100"]))
    for ax, name in zip(axes, DSETS):
        for mname in lc_models:
            s_ = lc[(lc.dataset == name) & (lc.model == mname)]
            ax.plot(s_.n_train, s_.test_acc, color=colors[mname], marker="o", ms=4, label=mname)
        ax.set_title(f"{name}: learning curves")
        ax.set_xlabel("training-set size")
    axes[1].legend(loc="lower right", fontsize=7.5)  # in the dset0 panel it would cover the logistic curve
    axes[0].set_ylabel("held-out accuracy (20 splits)")
    fig.tight_layout()
    fig.savefig(HERE / "problem3_learning_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
