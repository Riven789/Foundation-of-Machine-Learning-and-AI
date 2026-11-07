"""
Homework 2, Problem 1: Heteroscedastic regression for NYC taxi trip durations.

We fit an MLP g: R^7 -> R^2 whose two outputs are mapped to a conditional mean
and a conditional standard deviation

    f_mu(x)    = g_1(x)                       (identity link)
    f_sigma(x) = softplus(g_2(x)) + eps       (softplus link, keeps sigma > 0)

and train it with the Gaussian negative log-likelihood

    L = 1/n sum_i [ log f_sigma(x_i) + (y_i - f_mu(x_i))^2 / (2 f_sigma(x_i)^2) ] + const.

Baselines (all evaluated with the same Gaussian likelihood so they are comparable):
  1. Constant:            mu = mean(y_train), sigma = std(y_train)
  2. Linear + const sigma: OLS mean, sigma = std of training residuals
  3. Homoscedastic MLP:    same MLP for the mean, but a single learned sigma
                           (isolates the benefit of modelling sigma(x))

Then (d)/(e): P(Y < 45 | x) = Phi((45 - f_mu(x)) / f_sigma(x)), evaluated on the test set
with Brier score, log-loss and accuracy at the 0.5 threshold.

Data: homework_2/taxi_trips.parquet (one level above this folder)
Usage:
    pip install numpy pandas pyarrow scipy matplotlib torch
    python problem1.py
"""

import copy
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

SEED = 0
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "taxi_trips.parquet"
FIG_DIR = HERE

TARGET = "trip_duration"
THRESHOLD = 45.0  # minutes
SIGMA_FLOOR = 1e-3  # in standardized-target units

# training hyper-parameters
HIDDEN = (256, 256, 128)
BATCH_SIZE = 1024
LR = 1e-3
MAX_EPOCHS = 60
PATIENCE = 6

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, torch.get_num_threads()))
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_and_split():
    df = pd.read_parquet(DATA)
    features = [c for c in df.columns if c != TARGET]
    X = df[features].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)

    # 70 / 15 / 15 random split
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_tr, n_va = int(0.70 * len(df)), int(0.15 * len(df))
    tr, va, te = idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]
    return features, (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def mlp_trunk(d_in, hidden):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers), d


class HeteroscedasticMLP(nn.Module):
    """g: R^p -> R^2, then (identity, softplus) link -> (mu, sigma)."""

    def __init__(self, d_in, hidden=HIDDEN):
        super().__init__()
        self.trunk, d = mlp_trunk(d_in, hidden)
        self.head = nn.Linear(d, 2)
        # start with sigma ~ 1 (= std of the standardized target)
        with torch.no_grad():
            self.head.bias[1] = float(np.log(np.expm1(1.0)))

    def forward(self, x):
        g = self.head(self.trunk(x))
        mu = g[:, 0]
        sigma = F.softplus(g[:, 1]) + SIGMA_FLOOR
        return mu, sigma


class HomoscedasticMLP(nn.Module):
    """Same mean network, but one global (learned) sigma."""

    def __init__(self, d_in, hidden=HIDDEN):
        super().__init__()
        self.trunk, d = mlp_trunk(d_in, hidden)
        self.head = nn.Linear(d, 1)
        self.raw_sigma = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))

    def forward(self, x):
        mu = self.head(self.trunk(x)).squeeze(-1)
        sigma = (F.softplus(self.raw_sigma) + SIGMA_FLOOR).expand_as(mu)
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    """Per-example Gaussian NLL, including the 0.5*log(2*pi) constant."""
    return torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + 0.5 * np.log(2 * np.pi)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, X, batch=65536):
    model.eval()
    mus, sigmas = [], []
    for i in range(0, len(X), batch):
        mu, s = model(X[i : i + batch])
        mus.append(mu)
        sigmas.append(s)
    return torch.cat(mus), torch.cat(sigmas)


def train(model, train_t, val_t, name):
    Xtr, ytr = train_t
    Xva, yva = val_t
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    best_val, best_state, bad = np.inf, None, 0
    hist = {"train": [], "val": []}
    n = len(Xtr)
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH_SIZE):
            b = perm[i : i + BATCH_SIZE]
            mu, s = model(Xtr[b])
            loss = gaussian_nll(ytr[b], mu, s).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        tr_loss = tot / n

        mu, s = predict(model, Xva)
        va_loss = gaussian_nll(yva, mu, s).mean().item()
        sched.step(va_loss)
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        print(
            f"  [{name}] epoch {epoch:2d}  train NLL {tr_loss:.4f}  val NLL {va_loss:.4f}"
            f"  lr {opt.param_groups[0]['lr']:.1e}  ({time.time() - t0:.1f}s)"
        )

        if va_loss < best_val - 1e-4:
            best_val, best_state, bad = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  [{name}] early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return hist


# --------------------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------------------
def regression_metrics(y, mu, sigma):
    z = (y - mu) / sigma
    nll = np.mean(np.log(sigma) + 0.5 * z**2 + 0.5 * np.log(2 * np.pi))
    out = {
        "RMSE": np.sqrt(np.mean((y - mu) ** 2)),
        "MAE": np.mean(np.abs(y - mu)),
        "Gaussian NLL": nll,
    }
    for level in (0.5, 0.9, 0.95):
        q = stats.norm.ppf(0.5 + level / 2)
        out[f"{int(level * 100)}% PI coverage"] = np.mean(np.abs(z) <= q)
    return out


def prob_below(mu, sigma, t=THRESHOLD):
    """(d): P(Y < t | x) = Phi((t - mu(x)) / sigma(x)) under Y|x ~ N(mu, sigma^2)."""
    return stats.norm.cdf((t - mu) / sigma)


def roc_auc(label, score):
    """AUC via the Mann-Whitney U statistic (ties get average ranks)."""
    ranks = stats.rankdata(score)
    n_pos, n_neg = label.sum(), len(label) - label.sum()
    return (ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def classification_metrics(label, p):
    p_clip = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "Brier": np.mean((p - label) ** 2),
        "Log-loss": -np.mean(label * np.log(p_clip) + (1 - label) * np.log(1 - p_clip)),
        "Accuracy@0.5": np.mean((p >= 0.5) == label),
        "AUC": roc_auc(label, p),
    }


def plot_reliability(label, probs, path):
    """Reliability diagram for P(Y >= 45) (the rare event) -- more informative than P(Y<45)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.array([0, 1e-5, 1e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 1.0])
    for name, p in probs.items():
        q = np.clip(1 - p, 1e-6, 1)  # prob of a long trip (clipped for log-scale plotting)
        pos = 1 - label
        which = np.digitize(q, bins[1:-1])
        xs, ys = [], []
        for b in range(len(bins) - 1):
            m = which == b
            if m.sum() >= 50:
                xs.append(q[m].mean())
                ys.append(pos[m].mean())
        axes[0].plot(xs, ys, "o-", label=name)
        axes[1].hist(q, bins=np.logspace(-6, 0, 50), histtype="step", label=name, lw=1.5)
    lim = [1e-6, 1]
    axes[0].plot(lim, lim, "k--", lw=1, label="perfect calibration")
    axes[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim,
                xlabel="predicted P(trip >= 45 min)", ylabel="observed frequency",
                title="Reliability diagram (test set)")
    axes[0].legend(fontsize=8)
    axes[1].set(xscale="log", yscale="log", xlabel="predicted P(trip >= 45 min), clipped at 1e-6",
                ylabel="count",
                title="Distribution of predicted probabilities")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sigma(y, mu, sigma, path):
    """Does the predicted sigma track the actual error size?"""
    abs_res = np.abs(y - mu)
    edges = np.quantile(sigma, np.linspace(0, 1, 21))
    which = np.clip(np.digitize(sigma, edges[1:-1]), 0, 19)
    s_mean = np.array([sigma[which == b].mean() for b in range(20)])
    rmse = np.array([np.sqrt(np.mean(abs_res[which == b] ** 2)) for b in range(20)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(s_mean, rmse, "o-")
    m = max(s_mean.max(), rmse.max())
    axes[0].plot([0, m], [0, m], "k--", lw=1)
    axes[0].set(xlabel="predicted sigma (binned, minutes)", ylabel="empirical RMSE in bin (minutes)",
                title="Predicted vs. realised noise level (test)")
    sub = np.random.default_rng(SEED).choice(len(y), 5000, replace=False)
    axes[1].scatter(mu[sub], sigma[sub], s=3, alpha=0.4)
    axes[1].set(xlabel="predicted mean (minutes)", ylabel="predicted sigma (minutes)",
                title="sigma(x) vs mu(x) (5000 test points)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(hists, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, h in hists.items():
        ep = np.arange(1, len(h["train"]) + 1)
        ax.plot(ep, h["train"], "--", label=f"{name} train")
        ax.plot(ep, h["val"], "-", label=f"{name} val")
    ax.set(xlabel="epoch", ylabel="Gaussian NLL (standardized target)", title="Training curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    features, (Xtr, ytr), (Xva, yva), (Xte, yte) = load_and_split()
    print(f"features: {features}")
    print(f"train/val/test sizes: {len(ytr)}, {len(yva)}, {len(yte)}")
    print(f"fraction of trips < {THRESHOLD:.0f} min (train): {(ytr < THRESHOLD).mean():.4f}")

    preds = {}  # name -> (mu, sigma) on the test set, in minutes

    # ---- Baseline 1: constant Gaussian ----
    preds["Constant"] = (np.full_like(yte, ytr.mean()), np.full_like(yte, ytr.std()))

    # ---- Baseline 2: linear regression + constant sigma ----
    add1 = lambda X: np.hstack([np.ones((len(X), 1)), X.astype(np.float64)])
    beta, *_ = np.linalg.lstsq(add1(Xtr), ytr.astype(np.float64), rcond=None)
    sig_lin = np.std(ytr - add1(Xtr) @ beta)
    preds["Linear + const sigma"] = ((add1(Xte) @ beta).astype(np.float32), np.full_like(yte, sig_lin))

    # ---- MLPs: standardize the target for stable optimisation ----
    y_m, y_s = float(ytr.mean()), float(ytr.std())
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    tr_t = (to_t(Xtr), to_t((ytr - y_m) / y_s))
    va_t = (to_t(Xva), to_t((yva - y_m) / y_s))
    Xte_t = to_t(Xte)

    hists = {}
    for name, cls in [("Homoscedastic MLP", HomoscedasticMLP),
                      ("Heteroscedastic MLP", HeteroscedasticMLP)]:
        print(f"\nTraining {name}")
        torch.manual_seed(SEED)
        model = cls(Xtr.shape[1]).to(DEVICE)
        hists[name] = train(model, tr_t, va_t, name)
        mu, s = predict(model, Xte_t)
        # undo the standardization: mu = m + s*mu', sigma = s*sigma'
        preds[name] = (mu.cpu().numpy() * y_s + y_m, s.cpu().numpy() * y_s)

    # ---- (c) regression results ----
    reg = pd.DataFrame({k: regression_metrics(yte, *v) for k, v in preds.items()}).T
    print("\n=== (c) Test-set regression metrics (minutes; NLL in nats) ===")
    print(reg.to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- (d)/(e) probability that trip < 45 min ----
    label = (yte < THRESHOLD).astype(int)
    probs = {k: prob_below(*v) for k, v in preds.items()}
    cls_rows = {"Majority class (train freq.)": classification_metrics(
        label, np.full(len(yte), (ytr < THRESHOLD).mean()))}
    cls_rows.update({k: classification_metrics(label, p) for k, p in probs.items()})
    cls = pd.DataFrame(cls_rows).T
    print(f"\n=== (e) Test-set metrics for P(trip < {THRESHOLD:.0f} min) ===")
    print(f"test positive rate (trip < 45): {label.mean():.4f}   (n = {len(label)})")
    print(cls.to_string(float_format=lambda v: f"{v:.5f}"))

    # number of long trips actually caught by thresholding
    for k, p in probs.items():
        pred_long = p < 0.5
        tp = np.sum(pred_long & (label == 0))
        print(f"  {k:22s}: predicted long {pred_long.sum():5d}, correctly {tp:4d} "
              f"of {np.sum(label == 0)} actual long trips")

    reg.to_csv(HERE / "problem1_regression_results.csv")
    cls.to_csv(HERE / "problem1_classification_results.csv")

    plot_curves(hists, FIG_DIR / "problem1_training_curves.png")
    plot_reliability(label, probs, FIG_DIR / "problem1_reliability.png")
    plot_sigma(yte, *preds["Heteroscedastic MLP"], FIG_DIR / "problem1_sigma.png")
    print(f"\nSaved tables to {HERE} and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
