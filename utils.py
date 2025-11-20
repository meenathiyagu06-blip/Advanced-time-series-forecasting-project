import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def load_prophet_csv(path):
    df = pd.read_csv(path, parse_dates=['ds'])
    return df

def create_windows(values, input_len, output_len):
    # values: 1D or 2D numpy (T,) or (T, features)
    T = values.shape[0]
    X, Y = [], []
    for i in range(T - input_len - output_len + 1):
        X.append(values[i:i+input_len])
        Y.append(values[i+input_len:i+input_len+output_len])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def train_val_test_split(df, val_frac=0.1, test_frac=0.1):
    T = len(df)
    test_n = int(T * test_frac)
    val_n = int(T * val_frac)
    train = df.iloc[:T - val_n - test_n].reset_index(drop=True)
    val = df.iloc[T - val_n - test_n:T - test_n].reset_index(drop=True)
    test = df.iloc[T - test_n:].reset_index(drop=True)
    return train, val, test

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def plot_series(ds, y, prophet_pred=None, hybrid_pred=None, out_path=None):
    plt.figure(figsize=(12,4))
    plt.plot(ds, y, label='observed')
    if prophet_pred is not None:
        plt.plot(ds, prophet_pred, label='prophet')
    if hybrid_pred is not None:
        plt.plot(ds, hybrid_pred, label='hybrid (prophet+nn)')
    plt.legend()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path)
    else:
        plt.show()
