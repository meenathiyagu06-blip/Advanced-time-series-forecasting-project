"""Generate synthetic univariate series formatted for Prophet (ds, y)."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse, os

def generate(n_steps=1500, freq_minutes=60, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    # trend
    trend = 0.0008 * t
    # seasonal: daily + weekly like behavior (using minute frequency, but synthetic)
    daily = 2.0 * np.sin(2 * np.pi * t / 24)
    weekly = 1.0 * np.sin(2 * np.pi * t / (24*7))
    noise = rng.normal(scale=0.6, size=n_steps)
    y = 10.0 + trend + daily + weekly + noise
    start = datetime(2020,1,1)
    timestamps = [start + timedelta(minutes=freq_minutes*i) for i in range(n_steps)]
    df = pd.DataFrame({'ds': timestamps, 'y': y})
    return df

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/synthetic.csv')
    p.add_argument('--n_steps', type=int, default=1500)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df = generate(n_steps=args.n_steps)
    df.to_csv(args.out, index=False)
    print(f"Saved synthetic data to {args.out}")
