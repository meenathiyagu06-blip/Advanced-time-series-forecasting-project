# Advanced Time Series Forecasting — Prophet + Neural Network Hybrid

This project demonstrates a hybrid forecasting pipeline that combines
Prophet (for trend & seasonality) with a neural network (for modeling residuals).
The approach:
1. Fit Prophet on the historical series to capture trend and seasonality.
2. Compute residuals = (observed - prophet_fit).
3. Train a neural network (LSTM/MLP) on residual windows to predict future residuals.
4. Combine Prophet's forecast + predicted residuals = final forecast.

## Files
- `data_generator.py` — create synthetic time series suitable for Prophet.
- `prophet_nn_model.py` — utilities to fit Prophet and the PyTorch residual model.
- `train.py` — end-to-end training, evaluation and saving of artifacts.
- `utils.py` — data loading, windowing, plotting and metrics.
- `notebook_example.py` — quick runnable script demonstrating the workflow.
- `requirements.txt` — Python dependencies.
- `EXAMPLE_RUN.md` — short run instructions.
- `LICENSE` — MIT license.

## Quick start
1. Create venv and install: `pip install -r requirements.txt`
2. Generate data: `python data_generator.py --out data/synthetic.csv --n_steps 1500`
3. Train hybrid model: `python train.py --data data/synthetic.csv --save_dir outputs --input_len 56 --output_len 24 --epochs 10`
4. Check `outputs/` for model state, Prophet model, and plots.

## Notes
- This pipeline is a strong baseline: Prophet handles deterministic components, NN models remaining complex residual dynamics.
- For production, consider cross-validation, hyperparameter search, and more advanced residual models (Transformer, TCN).
