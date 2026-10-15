

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / 'data' / 'ale.csv'


class LinearModel(nn.Module):
    """define parameters w and b, input (n,p), output (n,)"""

    def __init__(self, n_features, initial_bias):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_features, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float64))

    def forward(self, x):
        return x @ self.w + self.b


def mse_loss(prediction, y):
    """(a)take average of squared errors over the batch."""
    return ((y - prediction) ** 2).mean()


def student_t_loss(prediction, y):
    """(b)take average of negative log-likelihood over the batch."""
    distribution = torch.distributions.StudentT(
        df=5.0, loc=prediction, scale=1.0
    )
    return -distribution.log_prob(y).mean()


def fit_sgd(x, y, loss_fn, epochs=2000, batch_size=32, lr=0.02, seed=42):
    """Mini-batch SGD randomize the training samples each epoch, with a decreasing learning rate.

    Two models are initialized with the same parameters and batch order for fair comparison.
    The bias is initialized to the mean of the training set y, and weights are initialized to zero.
    No test set is used to select hyperparameters, learning rate, epochs, or models.
    """
    x_tensor = torch.as_tensor(x, dtype=torch.float64)
    y_tensor = torch.as_tensor(y, dtype=torch.float64)
    model = LinearModel(x.shape[1], float(y.mean()))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history = []

    model.train()
    for epoch in range(epochs):
        # Decrease the learning rate over time to help convergence.
        optimizer.param_groups[0]['lr'] = lr / (1.0 + epoch / 100.0)
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), batch_size):
            indices = order[start:start + batch_size]
            prediction = model(x_tensor[indices])
            loss = loss_fn(prediction, y_tensor[indices])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss. Check data or reduce --lr.')
            optimizer.zero_grad()  # Clear old gradients from the last step.
            loss.backward()        # Compute gradients of the loss with respect to parameters w and b.
            optimizer.step()       # theta <- theta - learning_rate * gradient

        # Record the loss on the full training set after each epoch, without tracking gradients.
        with torch.no_grad():
            history.append(loss_fn(model(x_tensor), y_tensor).item())

    optimizer.zero_grad()
    loss_fn(model(x_tensor), y_tensor).backward()
    gradient_norm = torch.sqrt(sum(
        (parameter.grad ** 2).sum() for parameter in model.parameters()
    )).item()
    model.eval()
    return model, history, gradient_norm


def predict(model, x):
    """Make predictions with the trained model."""
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float64)).numpy()


def load_data(path, target, drop_columns):
    """Load the data from the previous problem"""
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(
            f'Target {target!r} not found. Available columns: {list(frame.columns)}'
        )
    if target in drop_columns:
        raise ValueError('The target cannot also be listed in --drop.')
    features = frame.drop(columns=[target] + drop_columns)
    if features.shape[1] == 0:
        raise ValueError('At least one feature is required.')
    if len(frame) < 5:
        raise ValueError('At least five observations are required for this split.')
    try:
        x = features.to_numpy(dtype=np.float64)
        y = frame[target].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            'Use numeric features from the previous problem; exclude identifiers '
            'with --drop or encode categorical features before running.'
        ) from error
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Missing/infinite values found. Check the source data first.')
    return x, y, features.columns.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA_PATH,
                        help='ALE CSV path (default: your supplied Mac path)')
    parser.add_argument('--target', default='ale', help='Target column (default: ale)')
    parser.add_argument('--drop', nargs='*', default=['sd_ale'],
                        help='Excluded columns (default: sd_ale; bare --drop includes it)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.02)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f'Data file not found: {args.data}. '
                     'Use --data /path/to/ale.csv to specify its location.')
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs, batch-size, and lr must be positive and finite.')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    x, y, feature_names = load_data(args.data, args.target, args.drop)

    # Step 1: Split the data into training and test sets, with a fixed random seed for reproducibility.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=args.seed, shuffle=True
    )

    # Step 2: Standardize the features using statistics from the training set only.
    # Adding a bias term does not change the expressiveness of the linear model.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Step 3: Train a baseline linear regression model using scikit-learn's LinearRegression.
    baseline = LinearRegression().fit(x_train, y_train)

    # Step 4: (a) Train a PyTorch linear model using MSE loss and mini-batch SGD.
    fit_options = dict(epochs=args.epochs, batch_size=args.batch_size,
                       lr=args.lr, seed=args.seed)
    mse_model, mse_history, mse_grad = fit_sgd(
        x_train, y_train, mse_loss, **fit_options
    )

    # Step 5: (b) Train a PyTorch linear model using Student-t NLL loss and mini-batch SGD.
    t_model, t_history, t_grad = fit_sgd(
        x_train, y_train, student_t_loss, **fit_options
    )

    # Step 6: Evaluate all three models on the test set and report MAE and difference from baseline.
    predictions = {
        'sklearn OLS baseline': baseline.predict(x_test),
        '(a) PyTorch MSE + SGD': predict(mse_model, x_test),
        '(b) PyTorch t(5) NLL + SGD': predict(t_model, x_test),
    }
    scores = {name: mean_absolute_error(y_test, pred)
              for name, pred in predictions.items()}
    baseline_mae = scores['sklearn OLS baseline']
    print(f'\nData: {args.data} | Target: {args.target}')
    print(f'Features: {feature_names}')
    print(f'Train: {len(y_train)} | Test: {len(y_test)} | Seed: {args.seed}')
    print(f'\n{"Model":<32} {"Test MAE":>12} {"MAE - baseline":>17}')
    for name, mae in scores.items():
        print(f'{name:<32} {mae:12.6f} {mae - baseline_mae:+17.6f}')
    print('Negative MAE difference = better than baseline on this test split.')

    # Step 7: Report training MSE and gradient norm for each PyTorch model, and compare to baseline OLS MSE.
    ols_mse = mean_squared_error(y_train, baseline.predict(x_train))
    print(f'\nOLS training MSE: {ols_mse:.8f}')
    print(f'(a) training MSE: {mse_history[-1]:.8f}')
    print(f'(a) training MSE gap vs OLS: {mse_history[-1] - ols_mse:.8f}')
    print(f'(a) full-training gradient norm: {mse_grad:.8f}')
    print(f'(b) training t NLL: {t_history[-1]:.8f}')
    print(f'(b) full-training gradient norm: {t_grad:.8f}')
    print('\nInterpretation: (a) should approach OLS when SGD converges. '
          '(b) downweights extreme residuals, but a test-MAE improvement is '
          'an empirical question. MSE and t NLL values are different objectives '
          'and should not be ranked against one another.')


if __name__ == '__main__':
    main()
