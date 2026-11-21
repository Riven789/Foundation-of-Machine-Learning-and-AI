"""
Homework 3, Problem 2: hyper-parameter selection and generalization.

  (b) simulation: k useless classifiers (true error 1/2) scored on one test set of size l.
      The minimum test error is far below 1/2 once log k is comparable to l.
  (c) white-wine quality regression (MSE). Random 70% train / 10% validation / 20% test split.
      Grid search with 600 hyper-parameter settings (more than the 490 validation points)
      for two model classes, RandomForestRegressor and KNeighborsRegressor. Every setting is
      fit on train and the "best" one is picked by validation MSE. Test MSE is then reported.
      The test MSE of every setting is also saved, but it is only used afterwards to study
      how validation risk relates to test risk. It is never used for selection.
      A repeated-split experiment (KNN grid, 20 random splits) measures the optimism of the
      selected validation MSE. It uses a fast, equivalent evaluation of the KNN grid
      (fast_knn_grid), checked against the sklearn grid on the main split.

Usage:
    pip install numpy pandas scikit-learn matplotlib
    python problem2.py
"""

import functools
import itertools
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "winequality-white.csv"
SEED = 0
print = functools.partial(print, flush=True)  # show progress when output is redirected to a file
N_REPEAT = 20  # random splits for the KNN optimism experiment

# plotting style (reference palette: blue, orange, aqua; muted ink; light grid)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 10,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------
def load():
    df = pd.read_csv(DATA, sep=";")
    X = df.drop(columns="quality").to_numpy(float)
    y = df["quality"].to_numpy(float)
    return df, X, y


def split_70_10_20(X, y, seed):
    """Random 70/10/20 split: test = 20%, then validation = 1/8 of the remaining 80% = 10%."""
    X_rest, X_te, y_rest, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_rest, y_rest, test_size=0.125, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def mse(y, p):
    return float(np.mean((y - p) ** 2))


# --------------------------------------------------------------------------------------------
# hyper-parameter grids (600 settings each, more than the 490 validation points)
# --------------------------------------------------------------------------------------------
RF_GRID = dict(
    n_estimators=[50, 150, 300],
    max_depth=[None, 5, 10, 20],
    min_samples_leaf=[1, 2, 4, 8, 16],
    max_features=[0.1, 0.3, 0.5, 0.7, 1.0],
    bootstrap=[True, False],
)
KNN_GRID = dict(
    scaler=["standard", "minmax"],
    n_neighbors=list(range(1, 51)),
    weights=["uniform", "distance"],
    metric=["manhattan", "euclidean", "chebyshev"],
)


def expand(grid):
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def make_rf(p):
    return RandomForestRegressor(random_state=SEED, n_jobs=-1, **p)


def make_knn(p):
    scaler = StandardScaler() if p["scaler"] == "standard" else MinMaxScaler()
    knn = KNeighborsRegressor(n_neighbors=p["n_neighbors"], weights=p["weights"], metric=p["metric"])
    return Pipeline([("scale", scaler), ("knn", knn)])


def grid_search(name, make, grid, split, verbose=True):
    X_tr, X_va, X_te, y_tr, y_va, y_te = split
    rows = []
    t0 = time.time()
    settings = expand(grid)
    for i, p in enumerate(settings):
        m = make(p).fit(X_tr, y_tr)
        rows.append({**p,
                     "train_mse": mse(y_tr, m.predict(X_tr)),
                     "val_mse": mse(y_va, m.predict(X_va)),
                     "test_mse": mse(y_te, m.predict(X_te))})
        if verbose and (i + 1) % 100 == 0:
            print(f"  {name}: {i + 1}/{len(settings)} settings ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res["model"] = name
    return res


def fast_knn_grid(split):
    """Validation / test MSE of all 600 KNN settings, computed with one neighbour query per
    (scaler, metric) pair instead of one model per setting. For each pair we find the 50 nearest
    training points of every validation / test point once. The prediction for n_neighbors = k is
    then built from the first k of them, with uniform weights or with sklearn's inverse-distance
    weights (if a query has a neighbour at distance 0, only the distance-0 neighbours get weight).
    main() checks on the main split that this gives the same numbers as fitting the 600
    sklearn models one at a time. They agree exactly except where the Chebyshev metric
    has exact distance ties, which can be broken differently."""
    X_tr, X_va, X_te, y_tr, y_va, y_te = split
    kmax = max(KNN_GRID["n_neighbors"])
    rows = []
    for scaler in KNN_GRID["scaler"]:
        sc = (StandardScaler() if scaler == "standard" else MinMaxScaler()).fit(X_tr)
        Z_tr, Z_va, Z_te = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)
        for metric in KNN_GRID["metric"]:
            nn = KNeighborsRegressor(n_neighbors=kmax, metric=metric, n_jobs=-1).fit(Z_tr, y_tr)
            res = {}
            for tag, Z, yy in [("val", Z_va, y_va), ("test", Z_te, y_te)]:
                dist, idx = nn.kneighbors(Z)
                res[tag] = (dist, y_tr[idx], yy)
            for k in KNN_GRID["n_neighbors"]:
                for w in KNN_GRID["weights"]:
                    out = {"scaler": scaler, "n_neighbors": k, "weights": w, "metric": metric}
                    for tag, (dist, ynb, yy) in res.items():
                        d, v = dist[:, :k], ynb[:, :k]
                        if w == "uniform":
                            pred = v.mean(axis=1)
                        else:
                            zero = d == 0
                            with np.errstate(divide="ignore"):
                                wt = np.where(zero.any(axis=1, keepdims=True), zero.astype(float), 1.0 / d)
                            pred = (wt * v).sum(axis=1) / wt.sum(axis=1)
                        out[f"{tag}_mse"] = mse(yy, pred)
                    rows.append(out)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# (b) simulation: selecting among k useless classifiers with one test set
# --------------------------------------------------------------------------------------------
def simulate_selection(l=100, ks=(1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000),
                       reps=200, seed=0):
    """Each 'model' errs on each test point independently w.p. 1/2 (true risk 1/2).
    Return the mean (over reps) of the smallest test error among k models.
    The test error of one model is Binomial(l, 1/2)/l, so we sample those directly."""
    rng = np.random.default_rng(seed)
    out = []
    for k in ks:
        mins = np.array([rng.binomial(l, 0.5, size=k).min() / l for _ in range(reps)])
        out.append({"k": k, "mean_min_test_error": mins.mean(), "sd": mins.std(),
                    "union_bound_rate": max(0.0, 0.5 - np.sqrt(np.log(k) / (2 * l)))})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------
def main():
    df, X, y = load()
    split = split_70_10_20(X, y, SEED)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split
    n_tr, n_va, n_te = len(y_tr), len(y_va), len(y_te)
    print(f"data: {df.shape}, quality range {y.min():.0f}-{y.max():.0f}")
    print(f"split: train {n_tr}, validation {n_va}, test {n_te}")

    # exact-duplicate rows (features + label) shared between train and val / test
    key = lambda A, b: set(map(tuple, np.column_stack([A, b]).round(6)))
    tr_keys = key(X_tr, y_tr)
    dup_va = np.mean([tuple(r) in tr_keys for r in np.column_stack([X_va, y_va]).round(6)])
    dup_te = np.mean([tuple(r) in tr_keys for r in np.column_stack([X_te, y_te]).round(6)])
    print(f"duplicate rows in file: {df.duplicated().sum()}; "
          f"val rows with an exact copy in train: {dup_va:.1%}, test rows: {dup_te:.1%}")

    base = y_tr.mean()
    baseline = {"val": mse(y_va, np.full(n_va, base)), "test": mse(y_te, np.full(n_te, base))}
    print(f"baseline (predict train mean {base:.3f}): val MSE {baseline['val']:.4f}, "
          f"test MSE {baseline['test']:.4f}")

    # ---------------- (c) grid search ----------------
    print(f"RF grid: {len(expand(RF_GRID))} settings, KNN grid: {len(expand(KNN_GRID))} settings")
    res_rf = grid_search("RandomForest", make_rf, RF_GRID, split)
    res_knn = grid_search("KNN", make_knn, KNN_GRID, split)
    res_rf.to_csv(HERE / "problem2_rf_grid.csv", index=False)
    res_knn.to_csv(HERE / "problem2_knn_grid.csv", index=False)

    summary = []
    for res, keys in [(res_rf, list(RF_GRID)), (res_knn, list(KNN_GRID))]:
        best = res.loc[res["val_mse"].idxmin()]
        oracle = res.loc[res["test_mse"].idxmin()]
        test_rank = int((res["test_mse"] < best["test_mse"]).sum()) + 1
        summary.append({
            "model": best["model"],
            "n_settings": len(res),
            "best_params": "; ".join(f"{k}={best[k]}" for k in keys),
            "train_mse": best["train_mse"],
            "val_mse_selected": best["val_mse"],
            "test_mse_selected": best["test_mse"],
            "test_minus_val": best["test_mse"] - best["val_mse"],
            "test_rank_of_selected": test_rank,
            "oracle_test_mse": oracle["test_mse"],
            "pearson_val_test": np.corrcoef(res["val_mse"], res["test_mse"])[0, 1],
            "spearman_val_test": res["val_mse"].rank().corr(res["test_mse"].rank()),
            "mean_test_minus_val_all": (res["test_mse"] - res["val_mse"]).mean(),
        })
    summary = pd.DataFrame(summary)

    # why is this validation set "easier"? label variance and the standard error of the
    # validation / test MSE of the two selected models
    print(f"label variance: train {y_tr.var():.3f}, validation {y_va.var():.3f}, test {y_te.var():.3f}")
    for res, make, keys in [(res_rf, make_rf, list(RF_GRID)), (res_knn, make_knn, list(KNN_GRID))]:
        best = res.loc[res["val_mse"].idxmin()]
        params = {k: best[k] for k in keys}
        if "max_depth" in params and pd.isna(params["max_depth"]):
            params["max_depth"] = None
        if "max_depth" in params and params["max_depth"] is not None:
            params["max_depth"] = int(params["max_depth"])
        for k in ("n_estimators", "min_samples_leaf", "n_neighbors"):
            if k in params:
                params[k] = int(params[k])
        if "bootstrap" in params:
            params["bootstrap"] = bool(params["bootstrap"])
        m = make(params).fit(X_tr, y_tr)
        e_va, e_te = (y_va - m.predict(X_va)) ** 2, (y_te - m.predict(X_te)) ** 2
        print(f"{best['model']} selected: val MSE {e_va.mean():.4f} (s.e. {e_va.std() / np.sqrt(n_va):.4f}), "
              f"test MSE {e_te.mean():.4f} (s.e. {e_te.std() / np.sqrt(n_te):.4f})")
    summary.to_csv(HERE / "problem2_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.max_colwidth", 120):
        print(summary.T)

    # Hoeffding + union bound (part (a)) evaluated for the validation set used here.
    # Predictions of RF / KNN are averages of training labels in [3, 9] and y is in [3, 9],
    # so the squared loss lies in [0, 36].
    B, delta, k = 36.0, 0.05, 600
    eps = B * np.sqrt(np.log(2 * k / delta) / (2 * n_va))
    print(f"bound (a) with B={B}, k={k}, l={n_va}, delta={delta}: eps = {eps:.3f}")

    # ---------------- repeated splits: optimism of the selected validation MSE (KNN) -----------
    check = fast_knn_grid(split).merge(res_knn, on=list(KNN_GRID), suffixes=("_fast", ""))
    dev = np.abs(check[["val_mse_fast", "test_mse_fast"]].to_numpy()
                 - check[["val_mse", "test_mse"]].to_numpy()).max(axis=1)
    cheb = (check["metric"] == "chebyshev").to_numpy()
    print(f"fast KNN grid vs sklearn grid on the main split: max |difference in MSE| = "
          f"{dev[~cheb].max():.1e} (manhattan/euclidean), {dev[cheb].max():.4f} (chebyshev)")
    # Manhattan / Euclidean agree exactly. Under the Chebyshev (max-coordinate) distance many
    # training points are exactly tied, and which tied point is returned depends on how many
    # neighbours are requested. Both answers are valid k-nearest-neighbour sets.
    assert dev[~cheb].max() < 1e-9
    rep = []
    for s in range(N_REPEAT):
        r = fast_knn_grid(split_70_10_20(X, y, 100 + s))
        b = r.loc[r["val_mse"].idxmin()]
        rep.append({"split_seed": 100 + s, "val_mse_selected": b["val_mse"],
                    "test_mse_selected": b["test_mse"], "oracle_test_mse": r["test_mse"].min(),
                    "n_neighbors": b["n_neighbors"], "weights": b["weights"],
                    "metric": b["metric"], "scaler": b["scaler"]})
        print(f"  repeat {s + 1}/{N_REPEAT}: val {b['val_mse']:.4f}  test {b['test_mse']:.4f}")
    rep = pd.DataFrame(rep)
    rep.to_csv(HERE / "problem2_knn_repeated_splits.csv", index=False)
    gap = rep["test_mse_selected"] - rep["val_mse_selected"]
    print(f"KNN over {N_REPEAT} splits: mean val(selected) {rep.val_mse_selected.mean():.4f}, "
          f"mean test(selected) {rep.test_mse_selected.mean():.4f}, mean gap {gap.mean():.4f} "
          f"(se {gap.std(ddof=1) / np.sqrt(N_REPEAT):.4f}), gap>0 in {(gap > 0).sum()}/{N_REPEAT}")

    # ---------------- (b) simulation ----------------
    sim = simulate_selection(l=100)
    sim.to_csv(HERE / "problem2_selection_simulation.csv", index=False)
    print(sim)

    make_figures(res_rf, res_knn, sim, n_va, n_te)


def make_figures(res_rf, res_knn, sim, n_va, n_te):
    # 1) validation MSE vs test MSE over all 600 settings, per model class
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    for ax, res, color in [(axes[0], res_rf, BLUE), (axes[1], res_knn, ORANGE)]:
        ax.scatter(res["val_mse"], res["test_mse"], s=10, color=color, alpha=0.55,
                   edgecolors="none", label="one setting")
        lo = min(res["val_mse"].min(), res["test_mse"].min()) - 0.01
        hi = max(res["val_mse"].max(), res["test_mse"].max()) + 0.01
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", label="test = validation")
        b = res.loc[res["val_mse"].idxmin()]
        ax.scatter([b["val_mse"]], [b["test_mse"]], s=70, marker="*", color=INK, zorder=5,
                   edgecolors="white", linewidths=0.8, label="selected (min val MSE)")
        ax.annotate(f"selected: val {b['val_mse']:.3f}, test {b['test_mse']:.3f}",
                    (b["val_mse"], b["test_mse"]), xytext=(40, -45), textcoords="offset points",
                    fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        r = np.corrcoef(res["val_mse"], res["test_mse"])[0, 1]
        ax.set_title(f"{res['model'].iloc[0]} (600 settings, Pearson r = {r:.2f})")
        ax.set_xlabel(f"validation MSE (l = {n_va})")
        ax.set_ylabel(f"test MSE ({n_te} points)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(HERE / "problem2_val_vs_test.png")
    plt.close(fig)

    # 2) KNN: validation / test MSE as a function of k for the selected scaler/metric/weights
    b = res_knn.loc[res_knn["val_mse"].idxmin()]
    sub = res_knn[(res_knn.scaler == b.scaler) & (res_knn.metric == b.metric)]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
    for ax, w in zip(axes, ["uniform", "distance"]):
        s = sub[sub.weights == w].sort_values("n_neighbors")
        ax.plot(s.n_neighbors, s.train_mse, color=AQUA, label="train")
        ax.plot(s.n_neighbors, s.val_mse, color=BLUE, label="validation")
        ax.plot(s.n_neighbors, s.test_mse, color=ORANGE, label="test")
        ax.set_title(f"weights = {w}")
        ax.set_xlabel("n_neighbors")
        ax.legend(loc="center right", fontsize=7.5)
    axes[0].set_ylabel("MSE")
    fig.suptitle(f"KNN with {b.scaler} scaling and {b.metric} distance", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "problem2_knn_curves.png")
    plt.close(fig)

    # 3) simulation for (b)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.text(1.1, 0.505, "true risk of every model = 0.5", fontsize=8, color=INK2, va="bottom")
    ax.plot(sim.k, sim.mean_min_test_error, color=BLUE, marker="o", ms=4,
            label="min test error over k models (mean of 200 runs)")
    ax.plot(sim.k, sim.union_bound_rate, color=ORANGE, label=r"$0.5-\sqrt{\log k/(2l)}$")
    ax.set_xscale("log")
    ax.set_xlabel("number of hyper-parameter settings k (log scale)")
    ax.set_ylabel("reported test error")
    ax.set_title("Selecting on the test set, l = 100")
    ax.set_ylim(0.15, 0.55)
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(HERE / "problem2_selection_simulation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
