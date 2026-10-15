"""
Homework 1, Problem 4: Are regression residuals approximately normal?

Fits several sklearn regressors to the ALE dataset (UCI id 844), predicting `ale`
from the first 4 variables (sd_ale is ignored), then checks residual normality
with QQ plots and scipy.stats.normaltest.

Residuals are computed two ways:
  * out-of-fold (5-fold CV): each residual comes from a model that did NOT see that
    point. This is the main analysis, because training residuals of flexible models
    can be shrunk toward 0 by overfitting.
  * in-sample (training): for comparison.

Data: homework_1/data/ale.csv (downloaded manually from https://archive.ics.uci.edu/dataset/844)

Usage:
    pip install -r requirements.txt
    python problem4.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, so the script also runs headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

SEED = 0
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "ale.csv"
FIG_DIR = HERE / "figures"
FEATURES = ["anchor_ratio", "trans_range", "node_density", "iterations"]
TARGET = "ale"


def load_data() -> pd.DataFrame:
    """Load the ALE data from data/ale.csv (committed to the repo)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the dataset from\n"
            "https://archive.ics.uci.edu/dataset/844 (click 'Download'), unzip it,\n"
            "and save the CSV inside as homework_1/data/ale.csv"
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        # Fall back to the column order described in the assignment:
        # the first 4 columns are the features and the next one is ale.
        print(f"Columns {missing} not found (got {list(df.columns)}); using column order instead.")
        renamed = dict(zip(df.columns[:5], FEATURES + [TARGET]))
        df = df.rename(columns=renamed)
    return df


def get_models() -> dict:
    """Models to compare. SVR and MLP are scale-sensitive, so they get a StandardScaler."""
    return {
        "Linear regression": LinearRegression(),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=10.0)),
        "Decision tree": DecisionTreeRegressor(max_depth=4, random_state=SEED),
        "Random forest": RandomForestRegressor(n_estimators=300, random_state=SEED),
        "Neural network (MLP)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, random_state=SEED),
        ),
    }


def main() -> None:
    df = load_data()
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    print(f"Loaded {len(df)} rows; predicting '{TARGET}' from {FEATURES}\n")

    FIG_DIR.mkdir(exist_ok=True)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = get_models()

    fig, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 8))
    rows = []

    for j, (name, model) in enumerate(models.items()):
        # Out-of-fold residuals: every prediction is made by a model not trained on that point.
        res_cv = y - cross_val_predict(model, X, y, cv=cv)
        # In-sample residuals: fit on all data, predict the same data.
        res_train = y - model.fit(X, y).predict(X)

        for i, (label, res) in enumerate([("out-of-fold", res_cv), ("training", res_train)]):
            stat, p = stats.normaltest(res)
            rows.append(
                {
                    "model": name,
                    "residuals": label,
                    "RMSE": np.sqrt(np.mean(res**2)),
                    "normaltest_stat": stat,
                    "p_value": p,
                    "skewness": stats.skew(res),
                    "excess_kurtosis": stats.kurtosis(res),
                }
            )
            ax = axes[i, j]
            stats.probplot(res, dist="norm", plot=ax)  # QQ plot against a normal distribution
            ax.set_title(f"{name}\n{label} residuals, p = {p:.3g}", fontsize=10)

    fig.tight_layout()
    fig_path = FIG_DIR / "qqplots.png"
    fig.savefig(fig_path, dpi=150)

    results = pd.DataFrame(rows)
    results.to_csv(HERE / "results.csv", index=False)
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        print(results.to_string(index=False))
    print(f"\nSaved QQ plots to {fig_path}")
    print(f"Saved table to {HERE / 'results.csv'}")
    print("normaltest H0: residuals are normal. Small p (e.g. < 0.05) => evidence against normality.")


if __name__ == "__main__":
    main()



#=============================================================================
# Results and interpretation
# (numbers below are from running this script on the real ALE data, n = 107)
# =============================================================================
#
# Setup
# -----
# Five models predict `ale` from anchor_ratio, trans_range, node_density, and iterations:
# linear regression, SVR (RBF kernel), a decision tree (max depth 4), a random forest,
# and an MLP neural network. For each model I computed the residuals r_i = y_i - y_hat_i
# in two ways:
#   * out-of-fold (5-fold CV): every residual comes from a model that did not see that point.
#     This is my main analysis, since it reflects how the model behaves on new data.
#   * training: fit on all data and predict the same data, for comparison only.
# Normality is checked with QQ plots (scipy.stats.probplot) and scipy.stats.normaltest
# (D'Agostino-Pearson test, H0: the residuals are normal). I also report skewness and
# excess kurtosis, which are both 0 for a normal distribution.

# Interpretation
# --------------
# 1. The residuals are not normal. For all five models, the out-of-fold residuals strongly
#    reject normality: K^2 ranges from 12.7 to 49.2, far above the 5% cutoff of about 6,
#    and p-values range from 1.7e-3 to 2e-11.
#
# 2. The deviation is a right skew with a heavy right tail. Skewness is positive (0.64 to
#    1.61) and excess kurtosis is positive (1.37 to 5.18) for every model. In the QQ plots,
#    the lower-left points stay close to the line, but the upper-right points bend upward.
#    So the models occasionally underpredict ALE by a lot, while large overpredictions
#    are rare.
#
# 3. This looks like a property of the data, not of one model. Five very different models
#    give the same pattern. A plausible reason is that ALE is an error measure and is always
#    positive, and positive quantities like this tend to be right-skewed, with a few unusually
#    large values that no model predicts well.
#
# 4. Training residuals can be misleading. The decision tree's training residuals give
#    p = 0.38, so on their own they would suggest normality. But its out-of-fold residuals
#    give p = 4.6e-7, and its RMSE nearly doubles from 0.144 (training) to 0.263
#    (out-of-fold). The tree overfits: it shrinks the training residuals and absorbs the
#    large outlying ones, so what is left looks more regular. The random forest shows the
#    same gap (training RMSE 0.070 vs. out-of-fold 0.208). This is why the test should use
#    residuals on data the model has not seen.
#
# 5. SVR is a special case. Its training-residual QQ plot has flat "plateaus" near -0.1 and
#    +0.1. SVR uses the epsilon-insensitive loss (epsilon = 0.1 by default): residuals inside
#    the +/- 0.1 tube cost nothing, so many residuals stop right at the tube's edge. SVR is not
#    trained with square error at all, so the "normal noise => square loss" argument does not
#    really apply to it. Its out-of-fold residuals are still non-normal (p = 0.0017).
#
# 6. Caveats. With n = 107, a single p-value should not be over-read, but here every model
#    gives a very small p-value and the QQ plots agree, so the conclusion is solid. Also, a
#    p-value only says whether there is evidence against normality; the skewness, kurtosis,
#    and QQ plots are what show how the residuals deviate.
#
# Conclusion
# ----------
# Square error loss follows from assuming Y = f(X) + eps with eps ~ N(0, sigma^2): under that
# assumption, maximizing the likelihood is the same as minimizing the sum of squared residuals.
# On this dataset the residuals are clearly right-skewed and heavy-tailed, so the
# "residuals are approximately normal" justification for square error loss does not hold.