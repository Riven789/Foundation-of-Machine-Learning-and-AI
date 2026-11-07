"""
Homework 2, Problem 2: Encoding high-cardinality categorical features.

Data: homework_2/retail.csv (one level above this folder)  (Quantity, StockCode, CustomerID)

Target / loss
-------------
Quantity is a positive integer with an extremely heavy right tail (median 6, max 80,995).
We model t = log(Quantity) and train with squared error on t. This is the Gaussian
likelihood for log(Quantity), i.e. a log-normal model for Quantity, so a single huge
order cannot dominate the loss. Metrics are reported on the log scale (RMSE, MAE, R^2)
and on the original scale (MAE of exp(prediction), the predicted median quantity).

Models (all share the same MLP head and training loop, so differences come only from the
input representation)
  * baselines: global mean of t; one-hot additive model (implemented as 1-d embeddings that
    are summed, which is exactly a linear model on the one-hot design matrix)
  (b) target encoding (TE) -> MLP     (also a linear model on TE for reference)
  (c) learned embeddings   -> MLP     (embedding dimension chosen on the validation set)
  (d) TE + embeddings      -> MLP
  (e) diagnostics: error by category frequency, seen vs unseen (customer, item) pairs,
      embedding geometry vs TE values.

Target encoding is implemented by hand (no category_encoders dependency):
    TE(c) = (n_c * mean_c + m * global_mean) / (n_c + m)
computed on the training set only. For the training rows themselves it is computed
out-of-fold (5 folds), so a row's own target never leaks into its feature.

Usage:
    pip install numpy pandas scipy matplotlib torch
    python problem2.py
"""

import copy
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 0
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "retail.csv"
FIG_DIR = HERE
CATS = ["StockCode", "CustomerID"]

TE_SMOOTHING = 20.0  # m in the smoothing formula
TE_FOLDS = 5
MIN_COUNT = 3  # categories seen < MIN_COUNT times in train share the "unknown" embedding
EMB_DIMS = [4, 8, 16, 32]
HIDDEN = (128, 64)
DROPOUT = 0.1
BATCH_SIZE = 1024
LR = 2e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 40
PATIENCE = 4

torch.manual_seed(SEED)
np.random.seed(SEED)


def log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------------------------
# (a) data, cardinality, split
# --------------------------------------------------------------------------------------
def load_and_split():
    df = pd.read_csv(DATA, dtype={"StockCode": str})
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    for c in CATS:
        df[c] = df[c].astype("category")
    df["t"] = np.log(df["Quantity"].astype(float))

    log("=== (a) cardinality ===")
    log(f"rows: {len(df)}")
    for c in CATS:
        counts = df[c].value_counts()
        log(f"{c:10s}: {df[c].nunique():5d} unique | rows per level: median {counts.median():.0f}, "
            f"min {counts.min()}, max {counts.max()} | levels with < 5 rows: {(counts < 5).sum()}")
    n_pairs = df.groupby(CATS, observed=True).ngroups
    log(f"unique (StockCode, CustomerID) pairs: {n_pairs}")
    n_onehot = sum(df[c].nunique() for c in CATS)
    log(f"one-hot width: {n_onehot} columns; dense float32 matrix = "
        f"{len(df) * n_onehot * 4 / 1e9:.1f} GB, sparse = {2 * len(df)} non-zeros")
    log(f"Quantity quantiles: " + ", ".join(
        f"{q:.3g}: {v:.0f}" for q, v in df["Quantity"].quantile([0.5, 0.9, 0.99, 0.999, 1]).items()))

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_tr, n_va = int(0.70 * len(df)), int(0.15 * len(df))
    parts = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    tr, va, te = (df.iloc[p].reset_index(drop=True) for p in parts)
    log(f"train/val/test: {len(tr)}, {len(va)}, {len(te)}")
    for c in CATS:
        seen = set(tr[c].unique())
        log(f"  test rows whose {c} is unseen in train: {(~te[c].isin(seen)).mean():.4f}")
    return tr, va, te


# --------------------------------------------------------------------------------------
# Target encoding (smoothed, out-of-fold on the training set)
# --------------------------------------------------------------------------------------
def te_map(keys, y, prior, m=TE_SMOOTHING):
    g = pd.DataFrame({"k": keys.astype(str), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * prior) / (g["count"] + m)


def target_encode(tr, others):
    """Returns TE features for train (out-of-fold) and for each frame in `others`."""
    prior = tr["t"].mean()
    out_tr = np.zeros((len(tr), len(CATS)), np.float32)
    folds = np.random.default_rng(SEED).integers(0, TE_FOLDS, len(tr))
    for j, c in enumerate(CATS):
        for f in range(TE_FOLDS):
            fit, app = folds != f, folds == f
            mp = te_map(tr.loc[fit, c], tr.loc[fit, "t"], prior)
            out_tr[app, j] = tr.loc[app, c].astype(str).map(mp).fillna(prior).to_numpy()
    outs = []
    for df in others:
        o = np.zeros((len(df), len(CATS)), np.float32)
        for j, c in enumerate(CATS):
            mp = te_map(tr[c], tr["t"], prior)
            o[:, j] = df[c].astype(str).map(mp).fillna(prior).to_numpy()
        outs.append(o)
    return out_tr, outs


def build_vocab(tr):
    """Index 0 = unknown / rare category; frequent training categories get 1..K."""
    vocabs = {}
    for c in CATS:
        counts = tr[c].astype(str).value_counts()
        keep = counts.index[counts >= MIN_COUNT]
        vocabs[c] = {k: i + 1 for i, k in enumerate(keep)}
    return vocabs


def to_index(df, vocabs):
    return np.stack([df[c].astype(str).map(vocabs[c]).fillna(0).astype(np.int64).to_numpy()
                     for c in CATS], axis=1)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def mlp(d_in, hidden=HIDDEN, dropout=DROPOUT):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class CatModel(nn.Module):
    """Embeddings (optional) + target-encoded features (optional) -> MLP."""

    def __init__(self, vocab_sizes, emb_dim=0, use_te=False, te_mean=None, te_std=None):
        super().__init__()
        self.emb_dim, self.use_te = emb_dim, use_te
        self.embs = nn.ModuleList([nn.Embedding(n, emb_dim) for n in vocab_sizes]) if emb_dim else None
        if self.embs is not None:
            for e in self.embs:
                nn.init.normal_(e.weight, std=0.05)
        d_in = emb_dim * len(vocab_sizes) + (len(CATS) if use_te else 0)
        self.net = mlp(d_in)
        self.register_buffer("te_mean", torch.as_tensor(te_mean if te_mean is not None else np.zeros(2)))
        self.register_buffer("te_std", torch.as_tensor(te_std if te_std is not None else np.ones(2)))

    def forward(self, idx, te):
        parts = []
        if self.embs is not None:
            parts += [e(idx[:, j]) for j, e in enumerate(self.embs)]
        if self.use_te:
            parts.append((te - self.te_mean) / self.te_std)
        return self.net(torch.cat(parts, dim=1)).squeeze(-1)


class AdditiveOneHot(nn.Module):
    """t_hat = b + w_stock[s] + w_customer[c]: a linear model on one-hot features,
    written with 1-d embeddings so we never materialize the one-hot matrix."""

    def __init__(self, vocab_sizes, bias):
        super().__init__()
        self.ws = nn.ModuleList([nn.Embedding(n, 1) for n in vocab_sizes])
        for w in self.ws:
            nn.init.zeros_(w.weight)
        self.b = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, idx, te):
        return self.b + sum(w(idx[:, j]).squeeze(-1) for j, w in enumerate(self.ws))


class LinearTE(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(len(CATS), 1)

    def forward(self, idx, te):
        return self.lin(te).squeeze(-1)


# --------------------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, data, batch=65536):
    model.eval()
    idx, te, _ = data
    return torch.cat([model(idx[i:i + batch], te[i:i + batch]) for i in range(0, len(idx), batch)])


def fit(model, train_d, val_d, name, weight_decay=WEIGHT_DECAY, lr=LR):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    idx, te, y = train_d
    n = len(y)
    best, best_state, bad, hist = np.inf, None, 0, {"train": [], "val": []}
    t_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            loss = ((model(idx[b], te[b]) - y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        va = ((predict(model, val_d) - val_d[2]) ** 2).mean().item()
        hist["train"].append(tot / n)
        hist["val"].append(va)
        if va < best - 1e-5:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    log(f"  [{name}] {epoch} epochs, best val MSE(log) {best:.4f}, "
        f"best epoch {int(np.argmin(hist['val'])) + 1} ({time.time() - t_start:.0f}s)")
    return model, hist, best


def metrics(t_true, t_pred):
    q_true, q_pred = np.exp(t_true), np.exp(t_pred)
    resid = t_true - t_pred
    return {
        "RMSE (log)": np.sqrt(np.mean(resid ** 2)),
        "MAE (log)": np.mean(np.abs(resid)),
        "R2 (log)": 1 - np.mean(resid ** 2) / np.var(t_true),
        "MAE (Quantity)": np.mean(np.abs(q_true - q_pred)),
        "MedAE (Quantity)": np.median(np.abs(q_true - q_pred)),
    }


# --------------------------------------------------------------------------------------
def main():
    tr, va, te = load_and_split()

    # ---- features ----
    te_tr, (te_va, te_te) = target_encode(tr, [va, te])
    vocabs = build_vocab(tr)
    vocab_sizes = [len(vocabs[c]) + 1 for c in CATS]
    log(f"embedding vocab sizes (incl. unknown index 0): {dict(zip(CATS, vocab_sizes))}")
    log(f"TE feature correlation with target (train, out-of-fold): "
        + ", ".join(f"{c}: {np.corrcoef(te_tr[:, j], tr['t'])[0, 1]:.3f}" for j, c in enumerate(CATS)))

    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    mk = lambda df, tef: (T(to_index(df, vocabs), torch.long), T(tef), T(df["t"].to_numpy()))
    D_tr, D_va, D_te = mk(tr, te_tr), mk(va, te_va), mk(te, te_te)
    te_mean, te_std = te_tr.mean(0), te_tr.std(0)
    y_te = te["t"].to_numpy()

    results, preds, hists = {}, {}, {}

    def record(name, model):
        p = predict(model, D_te).numpy()
        preds[name] = p
        results[name] = metrics(y_te, p)

    # ---- baselines ----
    preds["Global mean"] = np.full(len(te), tr["t"].mean())
    results["Global mean"] = metrics(y_te, preds["Global mean"])

    log("\n=== baseline: additive one-hot (linear fixed effects) ===")
    torch.manual_seed(SEED)
    m, hists["One-hot additive"], _ = fit(AdditiveOneHot(vocab_sizes, tr["t"].mean()), D_tr, D_va,
                                          "One-hot additive", weight_decay=0.0, lr=1e-2)
    record("One-hot additive (linear)", m)

    # ---- (b) target encoding ----
    log("\n=== (b) target encoding ===")
    torch.manual_seed(SEED)
    m, _, _ = fit(LinearTE(), D_tr, D_va, "TE -> linear", weight_decay=0.0, lr=1e-2)
    record("TE -> linear", m)
    torch.manual_seed(SEED)
    m, hists["TE -> MLP"], _ = fit(CatModel(vocab_sizes, 0, True, te_mean, te_std), D_tr, D_va, "TE -> MLP")
    record("TE -> MLP", m)

    # ---- (c) embeddings, sweep embedding dimension on validation ----
    log("\n=== (c) embeddings: sweep embedding dimension ===")
    sweep = {}
    for d in EMB_DIMS:
        torch.manual_seed(SEED)
        m, h, best = fit(CatModel(vocab_sizes, d), D_tr, D_va, f"Emb(d={d}) -> MLP")
        sweep[d] = (best, m, h)
    best_d = min(sweep, key=lambda d: sweep[d][0])
    log(f"best embedding dim on validation: {best_d}")
    emb_model = sweep[best_d][1]
    hists[f"Emb(d={best_d}) -> MLP"] = sweep[best_d][2]
    record(f"Embeddings (d={best_d}) -> MLP", emb_model)

    # ---- (d) TE + embeddings ----
    log("\n=== (d) TE + embeddings ===")
    torch.manual_seed(SEED)
    m, hists[f"TE + Emb(d={best_d}) -> MLP"], _ = fit(
        CatModel(vocab_sizes, best_d, True, te_mean, te_std), D_tr, D_va, f"TE + Emb(d={best_d}) -> MLP")
    combo_name = f"TE + Embeddings (d={best_d}) -> MLP"
    record(combo_name, m)

    res = pd.DataFrame(results).T
    log("\n=== Test-set results (target t = log Quantity) ===")
    log(res.to_string(float_format=lambda v: f"{v:.4f}"))
    res.to_csv(HERE / "problem2_results.csv")
    log("\nembedding-dimension sweep (val MSE on log scale): "
        + ", ".join(f"d={d}: {sweep[d][0]:.4f}" for d in EMB_DIMS))

    # ---- (e) diagnostics ----
    log("\n=== (e) diagnostics on the test set ===")
    main_models = ["Global mean", "One-hot additive (linear)", "TE -> MLP",
                   f"Embeddings (d={best_d}) -> MLP", combo_name]

    # (i) error by how often the item / customer appears in training
    for c in CATS:
        cnt = te[c].astype(str).map(tr[c].astype(str).value_counts()).fillna(0).to_numpy()
        groups = pd.cut(cnt, [-1, 0, 10, 50, 200, np.inf], labels=["0 (unseen)", "1-10", "11-50", "51-200", ">200"])
        tab = pd.DataFrame({k: pd.Series((y_te - preds[k]) ** 2).groupby(groups, observed=False).mean() ** 0.5
                            for k in main_models})
        tab.insert(0, "n_test", pd.Series(groups).value_counts().reindex(tab.index).to_numpy())
        log(f"\nRMSE(log) by number of training rows for the test row's {c}:")
        log(tab.to_string(float_format=lambda v: f"{v:.3f}"))
        tab.to_csv(HERE / f"problem2_rmse_by_{c}_frequency.csv")

    # (ii) is the (item, customer) pair itself seen in training? -> interaction effects
    pair_key = lambda df: df["StockCode"].astype(str) + "|" + df["CustomerID"].astype(str)
    pair_seen = pair_key(te).isin(set(pair_key(tr))).to_numpy()
    tab = pd.DataFrame({k: [np.sqrt(np.mean((y_te - preds[k])[pair_seen] ** 2)),
                            np.sqrt(np.mean((y_te - preds[k])[~pair_seen] ** 2))] for k in main_models},
                       index=["pair seen in train", "pair not seen"]).T
    log(f"\nRMSE(log) by whether the (StockCode, CustomerID) pair occurs in train "
        f"(seen: {pair_seen.mean():.3f} of test rows):")
    log(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    tab.to_csv(HERE / "problem2_rmse_by_pair_seen.csv")

    # (iii) how much interaction does each model use? Fit the best additive approximation
    #       to each model's own test predictions and measure what is left over.
    log("\nShare of each model's prediction variance NOT explained by an additive "
        "(item effect + customer effect) fit to its own predictions [test rows]:")
    for k in main_models[2:] + ["One-hot additive (linear)"]:
        log(f"  {k:35s}: {interaction_share(te, preds[k]):.3f}")

    # (iv) embedding geometry
    plot_embeddings(emb_model, vocabs, tr, FIG_DIR / "problem2_embeddings.png")
    plot_curves(hists, FIG_DIR / "problem2_training_curves.png")
    plot_sweep(sweep, FIG_DIR / "problem2_emb_dim_sweep.png")
    nearest_neighbours(emb_model, vocabs, tr)
    log(f"\nSaved results to {HERE} and figures to {FIG_DIR}")


def interaction_share(df, p, n_iter=30):
    """Backfitting of p ~ a[item] + b[customer]; returns residual var / var(p)."""
    s = df["StockCode"].astype(str).to_numpy()
    c = df["CustomerID"].astype(str).to_numpy()
    r = p - p.mean()
    a = np.zeros_like(r)
    b = np.zeros_like(r)
    for _ in range(n_iter):
        a = pd.Series(r - b).groupby(s).transform("mean").to_numpy()
        b = pd.Series(r - a).groupby(c).transform("mean").to_numpy()
    return np.var(r - a - b) / np.var(p)


def plot_embeddings(model, vocabs, tr, path):
    """PCA of the learned embeddings, coloured by each category's mean log-quantity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, c in enumerate(CATS):
        W = model.embs[j].weight.detach().numpy()[1:]  # drop the unknown row
        keys = list(vocabs[c].keys())
        g = tr.assign(k=tr[c].astype(str)).groupby("k")["t"].agg(["mean", "count"]).loc[keys]
        Wc = W - W.mean(0)
        U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
        Z = Wc @ Vt[:2].T
        evr = S[:2] ** 2 / np.sum(S ** 2)
        corr = [abs(np.corrcoef(Z[:, i], g["mean"])[0, 1]) for i in range(2)]
        sc = axes[j].scatter(Z[:, 0], Z[:, 1], c=g["mean"], s=np.clip(g["count"] / 20, 2, 40),
                             cmap="viridis", alpha=0.7, vmin=0, vmax=np.quantile(g["mean"], 0.99))
        axes[j].set(title=f"{c} embeddings (PCA)\n|corr(PC1, mean log-qty)| = {corr[0]:.2f}",
                    xlabel=f"PC1 ({evr[0]:.0%} var.)", ylabel=f"PC2 ({evr[1]:.0%} var.)")
        plt.colorbar(sc, ax=axes[j], label="training mean of log(Quantity)")
        log(f"  {c}: |corr(PC1, mean log qty)| = {corr[0]:.3f}, |corr(PC2, ...)| = {corr[1]:.3f}, "
            f"PC1+PC2 explain {evr.sum():.0%} of embedding variance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def nearest_neighbours(model, vocabs, tr, n_items=3, k=5):
    """Cosine nearest neighbours of a few popular items: do similar items cluster?"""
    W = model.embs[0].weight.detach().numpy()
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    inv = {i: s for s, i in vocabs["StockCode"].items()}
    mean_t = tr.assign(k=tr["StockCode"].astype(str)).groupby("k")["t"].mean()
    log("\nNearest neighbours (cosine) of the most frequent items in StockCode embedding space:")
    for s in tr["StockCode"].astype(str).value_counts().index[:n_items]:
        i = vocabs["StockCode"][s]
        sims = Wn[1:] @ Wn[i]
        order = np.argsort(-sims)[1:k + 1] + 1
        nb = ", ".join(f"{inv[o]} ({mean_t[inv[o]]:.2f})" for o in order)
        log(f"  {s} (mean log qty {mean_t[s]:.2f}) -> {nb}")


def plot_curves(hists, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, h) in enumerate(hists.items()):
        ep = np.arange(1, len(h["train"]) + 1)
        col = f"C{i}"
        ax.plot(ep, h["train"], "--", color=col)
        ax.plot(ep, h["val"], "-", color=col, label=name)
    ax.set(xlabel="epoch", ylabel="MSE on log(Quantity)", title="Training (dashed) and validation (solid) loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(sweep, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ds = sorted(sweep)
    ax.plot(ds, [sweep[d][0] for d in ds], "o-")
    ax.set(xscale="log", xticks=ds, xticklabels=ds, xlabel="embedding dimension",
           ylabel="best validation MSE (log)", title="Embedding dimension sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
