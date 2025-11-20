"""Train Prophet + NN hybrid model.

Workflow:
1. Load data (ds,y).
2. Split into train/val/test (chronological).
3. Fit Prophet on train; get in-sample fit and forecast for full timeline.
4. Compute residuals on train/val/test relative to Prophet predictions.
5. Create windows of residuals and train NN to predict next output_len residuals from input_len residuals.
6. At evaluation time, combine Prophet forecast + NN residual prediction.
"""
import argparse, os, numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from prophet_nn_model import fit_prophet, prophet_forecast, compute_residuals, ResidualMLP
from utils import load_prophet_csv, train_val_test_split, create_windows, mae, rmse, plot_series
from tqdm import trange

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/synthetic.csv')
    p.add_argument('--input_len', type=int, default=56)
    p.add_argument('--output_len', type=int, default=24)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save_dir', default='outputs')
    p.add_argument('--freq', default='H')  # hourly by default for make_future_dataframe
    p.add_argument('--evaluate_only', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    df = load_prophet_csv(args.data)
    train_df, val_df, test_df = train_val_test_split(df, val_frac=0.1, test_frac=0.1)

    # Fit Prophet on train
    print('Fitting Prophet on train set...')
    prophet = fit_prophet(train_df)
    # Forecast for entire timeline (train+val+test)
    total_periods = len(df) - len(train_df)
    forecast_full = prophet_forecast(prophet, df, periods=0, freq=args.freq)  # returns fitted yhat for historical ds
    # compute residuals across historical ds
    res_df = compute_residuals(df, forecast_full)
    # join residuals back with ds and y
    joined = df.merge(res_df, on='ds', how='left')

    # prepare windows on residuals using only train period for training NN
    residuals = joined['residual'].values
    train_n = len(train_df)
    train_res = residuals[:train_n]
    val_res = residuals[train_n:train_n + len(val_df)]
    test_res = residuals[train_n + len(val_df):]

    # create windows
    X_train, Y_train = create_windows(train_res, args.input_len, args.output_len)
    X_val, Y_val = create_windows(np.concatenate([train_res[-args.input_len:], val_res]) if len(val_res)>0 else train_res, args.input_len, args.output_len)
    X_test, Y_test = create_windows(np.concatenate([val_res[-args.input_len:] if len(val_res)>=args.input_len else np.concatenate([train_res[-(args.input_len-len(val_res)):], val_res]) , test_res]) , args.input_len, args.output_len)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResidualMLP(input_len=args.input_len, hidden=128, output_len=args.output_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train,dtype=torch.float32), torch.tensor(Y_train,dtype=torch.float32)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val,dtype=torch.float32), torch.tensor(Y_val,dtype=torch.float32)), batch_size=args.batch_size, shuffle=False)

    if args.evaluate_only:
        ckpt = os.path.join(args.save_dir, 'residual_model.pth')
        prophet_path = os.path.join(args.save_dir, 'prophet_model.pkl')
        model.load_state_dict(torch.load(ckpt, map_location=device))
        prophet = pd.read_pickle(prophet_path)
        print('Loaded artifacts. Evaluating...')
        evaluate_and_save(prophet, model, df, train_df, val_df, test_df, args, device)
        return

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            preds = model(xb).to(device)
            loss = loss_fn(preds, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # val
        model.eval()
        vloss = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                preds = model(xb)
                vloss.append(loss_fn(preds, yb).item())
        mean_v = float(np.mean(vloss)) if len(vloss)>0 else float('nan')
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={np.mean(losses):.6f} val_loss={mean_v:.6f}")
        if mean_v < best_val:
            best_val = mean_v
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'residual_model.pth'))
            # save prophet object
            pd.to_pickle(prophet, os.path.join(args.save_dir, 'prophet_model.pkl'))

    print('Training complete. Evaluating on test set...')
    evaluate_and_save(prophet, model, df, train_df, val_df, test_df, args, device)

def evaluate_and_save(prophet, residual_model, df, train_df, val_df, test_df, args, device):
    residual_model.eval()
    # Forecast future with prophet for full dataframe + horizon
    forecast_full = prophet_forecast(prophet, df, periods=0, freq=args.freq)
    merged = df.merge(forecast_full[['ds','yhat']], on='ds', how='left')
    # We'll produce hybrid forecast for test period by predicting residuals in sliding windows
    # assemble residuals series (observed - yhat)
    residuals = (merged['y'] - merged['yhat']).values
    # We'll produce predictions aligned to test indices
    train_n = len(train_df); val_n = len(val_df); test_n = len(test_df)
    start_idx = train_n + val_n  # index where test begins
    hybrid_preds = []
    prophet_preds = merged['yhat'].values[start_idx:start_idx+test_n]
    # sliding window forecasting for each step chunked by output_len
    for i in range(0, test_n - args.output_len + 1, args.output_len):
        # build input window using last input_len residuals available (from history + previous predictions)
        window_start = start_idx + i - args.input_len
        if window_start < 0:
            # pad with zeros if necessary
            pad = np.zeros(-window_start)
            window = np.concatenate([pad, residuals[:start_idx + i]])
        else:
            window = residuals[window_start:start_idx + i]
        # ensure correct length
        if len(window) < args.input_len:
            window = np.concatenate([np.zeros(args.input_len - len(window)), window])
        inp = torch.tensor(window.reshape(1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_res = residual_model(inp).cpu().numpy().reshape(-1)
        # append predicted residuals to residuals array for subsequent windows (autoregressive)
        residuals = np.concatenate([residuals, pred_res])
        # combine prophet forecast for that chunk with predicted residuals
        prophet_chunk = prophet_preds[i:i+args.output_len]
        hybrid_chunk = prophet_chunk + pred_res[:len(prophet_chunk)]
        hybrid_preds.extend(hybrid_chunk.tolist())

    hybrid_preds = np.array(hybrid_preds)[:test_n]
    true_test = df['y'].values[start_idx:start_idx+test_n]
    prophet_only = df.merge(forecast_full[['ds','yhat']], on='ds', how='left')['yhat'].values[start_idx:start_idx+test_n]

    # metrics
    from utils import mae, rmse, plot_series
    print('Prophet only MAE:', mae(true_test, prophet_only))
    print('Hybrid (prophet+NN) MAE:', mae(true_test, hybrid_preds))
    print('Prophet only RMSE:', rmse(true_test, prophet_only))
    print('Hybrid RMSE:', rmse(true_test, hybrid_preds))

    # save artifacts & plot
    np.save(os.path.join(args.save_dir, 'hybrid_preds.npy'), hybrid_preds)
    np.save(os.path.join(args.save_dir, 'prophet_only_preds.npy'), prophet_only)
    np.save(os.path.join(args.save_dir, 'true_test.npy'), true_test)
    plot_series(df['ds'].values[start_idx:start_idx+test_n], true_test, prophet_only, hybrid_preds, out_path=os.path.join(args.save_dir, 'test_plot.png'))
    print('Saved outputs to', args.save_dir)
    return

if __name__ == '__main__':
    main()
