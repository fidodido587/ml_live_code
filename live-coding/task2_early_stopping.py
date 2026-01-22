"""
Task 2 – Training Loop with Early Stopping

Implement a simple training function with early stopping:

    train_with_early_stopping(model, X_train, y_train, X_val, y_val, patience)

Assume:
    - `model` is a scikit-learn-like estimator with `fit` and `predict` methods.
    - You may re-fit the model each epoch, or use a model that supports partial_fit.
    - Validation loss is mean squared error (MSE).

Requirements:
    - Train in epochs.
    - After each epoch, compute validation MSE.
    - Track the best validation loss.
    - Stop training if the validation loss does not improve for `patience` consecutive epochs.
    - Print validation loss for each epoch.
    - Return the best model (i.e., the one with lowest validation loss).

You may use numpy and scikit-learn helpers if desired.
"""

from typing import Any, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


def train_with_early_stopping(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    patience: int = 3,
    max_epochs: int = 50,
) -> Tuple[Any, float]:
    """
    Train a model with early stopping based on validation MSE.

    Parameters
    ----------
    model : Any
        Estimator with fit() and predict() methods.
    X_train : np.ndarray
    y_train : np.ndarray
    X_val : np.ndarray
    y_val : np.ndarray
    patience : int, optional
        Number of epochs with no improvement after which training is stopped.
    max_epochs : int, optional
        Maximum number of training epochs.

    Returns
    -------
    best_model : Any
        Model corresponding to the best validation loss observed.
    best_val_loss : float
        The lowest validation MSE achieved.
    """
    # TODO: Implement early stopping logic here.
    raise NotImplementedError("Implement train_with_early_stopping() for Task 2.")


if __name__ == "__main__":
    # Optional: small smoke test (can be expanded by interviewer if needed)
    from sklearn.linear_model import SGDRegressor
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + rng.normal(scale=0.1, size=200)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

    base_model = SGDRegressor(random_state=42)
    best_model, best_loss = train_with_early_stopping(
        base_model, X_tr, y_tr, X_va, y_va, patience=5, max_epochs=50
    )
    print("Best validation MSE:", best_loss)
