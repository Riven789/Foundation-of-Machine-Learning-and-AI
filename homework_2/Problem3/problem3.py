"""
Homework 2, Problem 3: Random Kitchen Sinks (RKS) vs. a 1-hidden-layer MLP on a 2-d toy dataset.

  (a) visualise the training data (2-d scatter coloured by y, and a 3-d view)
  (b) RKS + ridge regression; sweep the kernel scale sigma (D = 2h = 1000, lambda = 1e-3 fixed)
  (c) 1-hidden-layer MLP (PyTorch); sweep the hidden width (lr, batch size, epochs fixed)
  (d) refit both with the best hyper-parameter, report validation MSE, write
      rks_test_predictions.csv / mlp_test_predictions.csv, and plot the fitted surfaces

Uses the provided dataloader.py and plotting.py (one level above this folder).

Usage:
    pip install numpy matplotlib torch
    python problem3.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from dataloader import HW2Dataset  # noqa: E402
from plotting import plot_surface_2d  # noqa: E402

SEED = 0
SEEDS = [0, 1, 2, 3, 4]  # repeat each sweep point over several seeds (randomness of RKS / NN init)

# RKS defaults
RKS_D = 1000  # total number of random features = 2h  (h = 500 sin/cos pairs)
RKS_LAMBDA = 1e-3
RKS_SIGMAS = np.logspace(np.log10(0.05), np.log10(10), 16)

# MLP defaults
MLP_WIDTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MLP_LR = 3e-3
MLP_BATCH = 64
MLP_EPOCHS = 600

torch.set_num_threads(4)


def log(msg=""):
    print(msg, flush=True)


def mse(a, b):
    return float(np.mean((np.asarray(a).ravel() - np.asarray(b).ravel()) ** 2))


# --------------------------------------------------------------------------------------
# Random Kitchen Sinks
# --------------------------------------------------------------------------------------
class RandomKitchenSinks:
    """z(x) = [sin(w_j^T x + b_j), cos(w_j^T x + b_j)]_{j=1..h},  w_j ~ N(0, I/sigma^2), b_j ~ U[0, 2pi],
    followed by ridge regression  min_beta 1/n ||y - beta_0 - Z beta||^2 + lambda ||beta||^2
    (the intercept beta_0 is not penalised)."""

    def __init__(self, D=RKS_D, sigma=1.0, ridge_lambda=RKS_LAMBDA, seed=0):
        assert D % 2 == 0
        self.D, self.sigma, self.lam, self.seed = D, sigma, ridge_lambda, seed

    def _init_params(self, d):
        rng = np.random.default_rng(self.seed)
        h = self.D // 2
        self.W = rng.normal(0.0, 1.0 / self.sigma, size=(d, h))  # columns w_j ~ N(0, I/sigma^2)
        self.b = rng.uniform(0.0, 2 * np.pi, size=h)

    def _features(self, X):
        proj = X @ self.W + self.b  # (n, h)
        return np.concatenate([np.sin(proj), np.cos(proj)], axis=1)  # (n, 2h = D)

    def fit(self, X, y):
        self._init_params(X.shape[1])
        Z = self._features(X)
        n = len(y)
        # centre to handle the unpenalised intercept
        self.z_mean, self.y_mean = Z.mean(0), y.mean()
        Zc, yc = Z - self.z_mean, y - self.y_mean
        # normal equations: (Zc^T Zc + n*lambda*I) beta = Zc^T yc
        A = Zc.T @ Zc + n * self.lam * np.eye(self.D)
        self.coef_ = np.linalg.solve(A, Zc.T @ yc)
        self.intercept_ = self.y_mean - self.z_mean @ self.coef_
        return self

    def predict(self, X):
        return self._features(X) @ self.coef_ + self.intercept_


def sweep_rks(ds):
    tr = np.zeros((len(RKS_SIGMAS), len(SEEDS)))
    va = np.zeros_like(tr)
    for i, s in enumerate(RKS_SIGMAS):
        for j, seed in enumerate(SEEDS):
            m = RandomKitchenSinks(sigma=s, seed=seed).fit(ds.train.X, ds.train.y)
            tr[i, j] = mse(m.predict(ds.train.X), ds.train.y)
            va[i, j] = mse(m.predict(ds.val.X), ds.val.y)
        log(f"  sigma = {s:7.4f}: train MSE {tr[i].mean():.4f} +- {tr[i].std():.4f}, "
            f"val MSE {va[i].mean():.4f} +- {va[i].std():.4f}")
    best = int(np.argmin(va.mean(1)))
    return tr, va, RKS_SIGMAS[best]


# --------------------------------------------------------------------------------------
# 1-hidden-layer MLP
# --------------------------------------------------------------------------------------
class MLP(nn.Module):
    """x -> Linear(2, H) -> ReLU -> Linear(H, 1)."""

    def __init__(self, d_in=2, d_hidden=128, lr=MLP_LR, batch_size=MLP_BATCH, epochs=MLP_EPOCHS, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lr, self.batch_size, self.epochs, self.seed = lr, batch_size, epochs, seed
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))

    def fit(self, X, y):
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float().view(-1, 1)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # cosine decay of the learning rate so the final iterate is not dominated by SGD noise
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        g = torch.Generator().manual_seed(self.seed)
        n = len(X)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                b = perm[i:i + self.batch_size]
                loss = ((self.net(X[b]) - y[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        self.net.eval()
        return self

    def forward(self, X):
        return self.net(X)

    def predict(self, X):
        with torch.no_grad():
            return self.net(torch.from_numpy(X).float()).numpy().ravel()


def sweep_mlp(ds):
    tr = np.zeros((len(MLP_WIDTHS), len(SEEDS)))
    va = np.zeros_like(tr)
    for i, h in enumerate(MLP_WIDTHS):
        for j, seed in enumerate(SEEDS):
            m = MLP(d_hidden=h, seed=seed).fit(ds.train.X, ds.train.y)
            tr[i, j] = mse(m.predict(ds.train.X), ds.train.y)
            va[i, j] = mse(m.predict(ds.val.X), ds.val.y)
        log(f"  width = {h:5d}: train MSE {tr[i].mean():.4f} +- {tr[i].std():.4f}, "
            f"val MSE {va[i].mean():.4f} +- {va[i].std():.4f}")
    best = int(np.argmin(va.mean(1)))
    return tr, va, MLP_WIDTHS[best]


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_data(ds, path):
    X, y = ds.train.X, ds.train.y
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 2, 1)
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=22, edgecolor="k", linewidth=0.2)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="x1", ylabel="x2", aspect="equal",
           title="Training data (colour = y)")
    plt.colorbar(sc, ax=ax, label="y")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_trisurf(X[:, 0], X[:, 1], y, cmap="viridis", alpha=0.85, linewidth=0)
    ax.scatter(X[:, 0], X[:, 1], y, c="k", s=3)
    ax.set(xlabel="x1", ylabel="x2", zlabel="y", title="Training data (3-d, triangulated)")
    ax.view_init(elev=30, azim=-60)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(xs, tr, va, best, xlabel, title, path, logx=True):
    """Two panels as required: training loss and validation loss vs. the hyper-parameter."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, L, name, col in [(axes[0], tr, "Training", "C0"), (axes[1], va, "Validation", "C1")]:
        m, s = L.mean(1), L.std(1)
        ax.plot(xs, m, "o-", color=col)
        ax.fill_between(xs, m - s, m + s, color=col, alpha=0.25, label=f"mean $\\pm$ sd over {L.shape[1]} seeds")
        ax.axvline(best, color="k", ls="--", lw=1, label=f"best (val) = {best:g}")
        ax.set(xlabel=xlabel, ylabel="MSE", title=f"{name} MSE vs. {title}", yscale="log")
        if logx:
            ax.set_xscale("log")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(ds, rks, mlp, rks_label, mlp_label, path, grid=200):
    xs = np.linspace(-3, 3, grid)
    X1, X2 = np.meshgrid(xs, xs)
    G = np.stack([X1.ravel(), X2.ravel()], axis=1)
    Zr = rks.predict(G).reshape(grid, grid)
    Zm = mlp.predict(G).reshape(grid, grid)
    vmin = min(ds.train.y.min(), Zr.min(), Zm.min())
    vmax = max(ds.train.y.max(), Zr.max(), Zm.max())
    levels = np.linspace(vmin, vmax, 31)

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    axes[0].scatter(ds.train.X[:, 0], ds.train.X[:, 1], c=ds.train.y, cmap="viridis",
                    vmin=vmin, vmax=vmax, s=22, edgecolor="k", linewidth=0.2)
    axes[0].set_title("Training data")
    axes[1].contourf(X1, X2, Zr, levels=levels, cmap="viridis")
    axes[1].set_title(f"RKS fit ({rks_label})")
    cf = axes[2].contourf(X1, X2, Zm, levels=levels, cmap="viridis")
    axes[2].set_title(f"MLP fit ({mlp_label})")
    for ax in axes[:3]:
        ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="x1", ylabel="x2", aspect="equal")
    fig.colorbar(cf, ax=axes[:3].tolist(), label="y / prediction", shrink=0.9)
    d = Zr - Zm
    lim = np.abs(d).max()
    cf2 = axes[3].contourf(X1, X2, d, levels=np.linspace(-lim, lim, 31), cmap="RdBu_r")
    axes[3].set(xlim=(-3, 3), ylim=(-3, 3), xlabel="x1", ylabel="x2", aspect="equal",
                title="RKS $-$ MLP")
    fig.colorbar(cf2, ax=axes[3], shrink=0.9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    ds = HW2Dataset.from_csv(str(ROOT / "train.csv"), str(ROOT / "val.csv"), str(ROOT / "test.csv"))
    log(f"train {ds.train.X.shape}, val {ds.val.X.shape}, test {ds.test.X.shape}")
    log(f"train y: mean {ds.train.y.mean():.3f}, var {ds.train.y.var():.3f}; val y var {ds.val.y.var():.3f}")
    log(f"constant baseline (train mean) val MSE: {mse(ds.train.y.mean(), ds.val.y):.4f}")

    # (a)
    plot_data(ds, HERE / "problem3_data.png")

    # (b)
    log(f"\n=== (b) RKS: sweep sigma  (D = {RKS_D}, lambda = {RKS_LAMBDA}) ===")
    tr, va, best_sigma = sweep_rks(ds)
    log(f"best sigma (mean val MSE over seeds): {best_sigma:.4f}")
    plot_sweep(RKS_SIGMAS, tr, va, round(best_sigma, 4), r"kernel scale $\sigma$", r"$\sigma$",
               HERE / "problem3_rks_sweep.png")
    np.savetxt(HERE / "problem3_rks_sweep.csv",
               np.column_stack([RKS_SIGMAS, tr.mean(1), tr.std(1), va.mean(1), va.std(1)]),
               delimiter=",", header="sigma,train_mse_mean,train_mse_sd,val_mse_mean,val_mse_sd", comments="")

    # (c)
    log(f"\n=== (c) MLP: sweep hidden width  (Adam lr {MLP_LR}, batch {MLP_BATCH}, "
        f"{MLP_EPOCHS} epochs, ReLU) ===")
    tr, va, best_width = sweep_mlp(ds)
    log(f"best hidden width (mean val MSE over seeds): {best_width}")
    plot_sweep(np.array(MLP_WIDTHS), tr, va, best_width, "hidden width H", "hidden width",
               HERE / "problem3_mlp_sweep.png")
    np.savetxt(HERE / "problem3_mlp_sweep.csv",
               np.column_stack([MLP_WIDTHS, tr.mean(1), tr.std(1), va.mean(1), va.std(1)]),
               delimiter=",", header="width,train_mse_mean,train_mse_sd,val_mse_mean,val_mse_sd", comments="")

    # (d) refit with the best hyper-parameters (seed 0) and compare
    log("\n=== (d) refit with best hyper-parameters ===")
    rks = RandomKitchenSinks(sigma=best_sigma, seed=SEED).fit(ds.train.X, ds.train.y)
    mlp = MLP(d_hidden=best_width, seed=SEED).fit(ds.train.X, ds.train.y)
    n_val = len(ds.val.y)
    for name, m in [("RKS", rks), ("MLP", mlp)]:
        r = (ds.val.y - m.predict(ds.val.X)) ** 2
        log(f"  {name}: train MSE {mse(m.predict(ds.train.X), ds.train.y):.4f}, "
            f"val MSE {r.mean():.4f} (SE over validation points {r.std(ddof=1) / np.sqrt(n_val):.4f})")

    # Uncertainty of the comparison. The +- in the sweeps only covers feature/init randomness;
    # the 200 validation points are themselves a sample. Both models are scored on the SAME
    # points, so the paired difference has a much smaller SE than either MSE on its own.
    r_rks = (ds.val.y - rks.predict(ds.val.X)) ** 2
    r_mlp = (ds.val.y - mlp.predict(ds.val.X)) ** 2
    diff = r_mlp - r_rks
    d_mean, d_se = diff.mean(), diff.std(ddof=1) / np.sqrt(n_val)
    se_unpaired = np.sqrt(r_rks.var(ddof=1) / n_val + r_mlp.var(ddof=1) / n_val)
    log(f"  paired difference MSE(MLP) - MSE(RKS) on validation: {d_mean:.4f} +- {d_se:.4f} (SE), "
        f"t = {d_mean / d_se:.2f}; unpaired SE would be {se_unpaired:.4f}")
    log(f"  correlation of squared errors (RKS vs MLP) across validation points: "
        f"{np.corrcoef(r_rks, r_mlp)[0, 1]:.3f}")
    # does the gap persist across random seeds of both models?
    gaps = []
    for seed in SEEDS:
        a = RandomKitchenSinks(sigma=best_sigma, seed=seed).fit(ds.train.X, ds.train.y)
        b = MLP(d_hidden=best_width, seed=seed).fit(ds.train.X, ds.train.y)
        gaps.append(mse(b.predict(ds.val.X), ds.val.y) - mse(a.predict(ds.val.X), ds.val.y))
    log(f"  same paired gap for seeds {SEEDS}: " + ", ".join(f"{g:.4f}" for g in gaps)
        + f"  (mean {np.mean(gaps):.4f})")
    n_params_mlp = sum(p.numel() for p in mlp.parameters())
    log(f"  # trainable parameters: RKS {RKS_D + 1} (linear head only), MLP {n_params_mlp}")

    np.savetxt(HERE / "rks_test_predictions.csv", rks.predict(ds.test.X), fmt="%.10f")
    np.savetxt(HERE / "mlp_test_predictions.csv", mlp.predict(ds.test.X), fmt="%.10f")
    log(f"  test predictions written ({len(ds.test.X)} rows each); "
        f"MSE between RKS and MLP test predictions: {mse(rks.predict(ds.test.X), mlp.predict(ds.test.X)):.4f}")

    # surfaces with the provided helper, plus a side-by-side comparison with the data
    plot_surface_2d(rks.predict, (-3, 3), (-3, 3), title=f"RKS surface (sigma = {best_sigma:.3g}, D = {RKS_D})",
                    filename=str(HERE / "problem3_rks_surface.png"))
    plt.close("all")
    plot_surface_2d(mlp, (-3, 3), (-3, 3), title=f"MLP surface (hidden width = {best_width})",
                    filename=str(HERE / "problem3_mlp_surface.png"))
    plt.close("all")
    plot_comparison(ds, rks, mlp, f"$\\sigma$={best_sigma:.3g}", f"H={best_width}",
                    HERE / "problem3_comparison.png")
    log(f"\nSaved outputs to {HERE}")


if __name__ == "__main__":
    main()
