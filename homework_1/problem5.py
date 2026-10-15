"""
Homework 1, Problem 5: models with custom losses via automatic differentiation (PyTorch).

(a) Logistic regression: f(x) = sigmoid(w^T x + b)
(b) Probit regression:   f(x) = Phi(w^T x + b), Phi = standard normal CDF

Both are trained by minimizing the same binary cross-entropy loss with mini-batch SGD
(torch.optim.SGD), and compared with sklearn's LogisticRegression as a baseline.

Usage:
    pip install -r requirements.txt
    python problem5.py
"""

import torch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 32
LR = 0.1          # SGD step size
BATCH_SIZE = 32   # samples per mini-batch
NUM_EPOCHS = 100  # passes over the training set
EPS = 1e-7        # clipping level for the predicted probability

STD_NORMAL = torch.distributions.Normal(0.0, 1.0)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_and_split():
    """80/20 split, then standardize using statistics from the training set only."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train)  # mean/std from training data only (no leakage)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


# ----------------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------------
def run_baseline(X_train, X_test, y_train, y_test):
    """sklearn logistic regression. Note: by default it adds L2 regularization (C=1.0)."""
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, test_acc


# ----------------------------------------------------------------------------
# Models: z = w^T x + b, then map z to a probability
# ----------------------------------------------------------------------------
def model_logistic(X, w, b):
    """(a) Logistic regression: f(x_i) = sigmoid(w^T x_i + b)."""
    return torch.sigmoid(X @ w + b)


def model_probit(X, w, b):
    """(b) Probit regression: f(x_i) = Phi(w^T x_i + b), Phi = standard normal CDF."""
    return STD_NORMAL.cdf(X @ w + b)


# ----------------------------------------------------------------------------
# Loss (shared by both models)
# ----------------------------------------------------------------------------
def bce_loss(y, f, eps=EPS):
    """Binary cross-entropy, averaged over samples. f is clipped to [eps, 1 - eps] so that
    log(f) and log(1 - f) never see 0. The loss only uses the probability f, so it works
    unchanged for any model that outputs a probability."""
    f = torch.clamp(f, eps, 1 - eps)
    return -torch.mean(y * torch.log(f) + (1 - y) * torch.log(1 - f))


# ----------------------------------------------------------------------------
# Training and evaluation (shared by both models)
# ----------------------------------------------------------------------------
def train(X_train, y_train, model_fn):
    """Minimize the BCE loss of `model_fn` with mini-batch SGD. Returns the learned (w, b)."""
    torch.manual_seed(SEED)  # same mini-batch order for every model, for a fair comparison

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    n, p = X.shape  # n training samples, p = 30 features

    # 31 parameters in total: one weight per feature plus one bias.
    # requires_grad=True tells PyTorch to track these so it can compute gradients for them.
    w = torch.zeros(p, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        perm = torch.randperm(n)  # new random order each epoch
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]

            optimizer.zero_grad()                          # 1. clear old gradients
            loss = bce_loss(y[idx], model_fn(X[idx], w, b))  # 2. forward pass on this batch
            loss.backward()                                # 3. autodiff: dL/dw and dL/db
            optimizer.step()                               # 4. theta <- theta - lr * grad

        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():  # only monitoring, no gradients needed
                full_loss = bce_loss(y, model_fn(X, w, b)).item()
            print(f"  epoch {epoch:3d}   training loss {full_loss:.4f}")

    return w.detach(), b.detach()


def predict(X, w, b, model_fn):
    """Predict class 1 when f(x) >= 0.5 (equivalently, when w^T x + b >= 0)."""
    with torch.no_grad():
        prob = model_fn(torch.tensor(X, dtype=torch.float32), w, b)
    return (prob >= 0.5).int().numpy(), prob


def clipped_fraction(prob, eps=EPS):
    """Fraction of predictions that fall in the clipped region (and so give zero gradient)."""
    return ((prob < eps) | (prob > 1 - eps)).float().mean().item()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    X_train, X_test, y_train, y_test = load_and_split()
    n_test = len(y_test)
    print(f"Training samples: {len(y_train)}, test samples: {n_test}\n")

    base_train, base_test = run_baseline(X_train, X_test, y_train, y_test)
    results = {"sklearn baseline": (base_train, base_test)}
    params, test_preds, clip = {}, {}, {}

    for name, model_fn in [("(a) logistic, ours", model_logistic),
                           ("(b) probit, ours", model_probit)]:
        print(f"Training {name} with SGD:")
        w, b = train(X_train, y_train, model_fn)
        train_pred, train_prob = predict(X_train, w, b, model_fn)
        test_pred, _ = predict(X_test, w, b, model_fn)
        results[name] = (accuracy_score(y_train, train_pred), accuracy_score(y_test, test_pred))
        params[name] = w
        test_preds[name] = test_pred
        clip[name] = clipped_fraction(train_prob)
        print()

    # Accuracy table
    print("                          train acc   test acc   correct on test")
    for name, (tr, te) in results.items():
        print(f"  {name:22s}  {tr:.4f}      {te:.4f}     {round(te * n_test):3d} / {n_test}")
    print(f"  (one test sample = {1 / n_test:.2%} accuracy)\n")

    # Extra diagnostics for comparing the two link functions
    logit_w, probit_w = params["(a) logistic, ours"], params["(b) probit, ours"]
    agree = (test_preds["(a) logistic, ours"] == test_preds["(b) probit, ours"]).mean()
    print("Logistic vs. probit:")
    print(f"  test predictions that agree:        {agree:.2%}")
    print(f"  ||w_logistic|| / ||w_probit||:      {(logit_w.norm() / probit_w.norm()).item():.3f}"
          "   (sigmoid(z) ~ Phi(z/1.7) suggests about 1.6-1.8)")
    for name, frac in clip.items():
        print(f"  training predictions clipped, {name}: {frac:.2%}")


if __name__ == "__main__":
    main()
