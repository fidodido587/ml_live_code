"""
Task 1 – Time-Series Feature Engineering

You are given a DataFrame with the following columns:

    timestamp | value
    ------------------
    2025-01-01 00:00:00 | 10.5
    2025-01-01 00:01:00 | 10.7
    ...

Timestamps may contain gaps (missing minutes).

Implement the function:

    create_time_windows(df, window_size, horizon)

Requirements:
    - Sort the data by timestamp.
    - Reindex to a continuous 1-minute frequency.
    - Forward-fill missing values.
    - For each time t, create a feature window of the previous `window_size` values.
    - The label y for that window should be the value at time t + horizon.
    - Do not leak future information into the features.
    - Return:
        X: numpy array, shape (n_samples, window_size)
        y: numpy array, shape (n_samples,)

You may use pandas and numpy.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def create_time_windows(
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
    freq: str = "1min",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding time-series windows from a timestamped univariate series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'timestamp' and 'value' columns.
    window_size : int
        Number of past steps to include in each feature window.
    horizon : int
        Number of steps ahead to predict the target value.
    freq : str, optional
        Frequency string for reindexing (default is "1min").

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, window_size).
    y : np.ndarray
        Target vector of shape (n_samples,).
    """
    # TODO: Implement the logic according to the docstring.
    raise NotImplementedError("Implement create_time_windows() for Task 1.")


if __name__ == "__main__":
    # Optional: simple test using the provided sample CSV
    df_sample = pd.read_csv("task1_sample_data.csv", parse_dates=["timestamp"])
    X, y = create_time_windows(df_sample, window_size=5, horizon=1)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
