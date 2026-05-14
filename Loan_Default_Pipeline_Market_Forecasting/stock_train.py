# import json
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import joblib

# BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# OUT = BASE_DIR / "output"
# OUT.mkdir(exist_ok=True)

# UNIVERSE_PATH = OUT / "synthetic_stock_universe.csv"
# DATA_PATH = OUT / "synthetic_stock_data.csv"
# MODEL_PATH = OUT / "stock_lstm_model.keras"
# SCALER_PATH = OUT / "stock_scaler.joblib"
# FEATURES_PATH = OUT / "stock_feature_columns.json"
# METRICS_PATH = OUT / "stock_metrics.json"
# FORECAST_PATH = OUT / "stock_forecast_output.csv"
# SCALER_META_PATH = OUT / "stock_scaler_meta.json"


# def create_stock_universe():
#     stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS", "ITC.NS", "TITAN.NS", "AXISBANK.NS"]
#     sectors = ["Energy", "IT", "Banking", "IT", "Banking", "Construction", "Banking", "FMCG", "Consumer", "Banking"]
#     beta = [1.15, 0.95, 1.05, 0.90, 1.00, 1.10, 1.20, 0.85, 0.98, 1.08]
#     rng = np.random.default_rng(7)
#     df = pd.DataFrame({
#         "symbol": stocks,
#         "sector": sectors,
#         "beta": beta,
#         "momentum": np.round(rng.uniform(0.2, 0.95, len(stocks)), 3),
#         "valuation": np.round(rng.uniform(0.15, 0.85, len(stocks)), 3),
#         "dividend_yield": np.round(rng.uniform(0.01, 0.035, len(stocks)), 4),
#         "pe_ratio": np.round(rng.uniform(12, 45, len(stocks)), 2),
#         "de_ratio": np.round(rng.uniform(0.05, 1.8, len(stocks)), 2),
#     })
#     df.to_csv(UNIVERSE_PATH, index=False)
#     return df


# def generate_stock_history(symbol, n_days=1500, seed=42):
#     seed_val = abs(hash(symbol)) % (2**32)
#     rng = np.random.default_rng(seed + seed_val)
#     dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
#     drift_map = {"RELIANCE.NS": 0.0005, "TCS.NS": 0.00045, "HDFCBANK.NS": 0.00042, "INFY.NS": 0.00048, "ICICIBANK.NS": 0.00044, "LT.NS": 0.00041, "SBIN.NS": 0.00047, "ITC.NS": 0.00033, "TITAN.NS": 0.00046, "AXISBANK.NS": 0.00043}
#     vol_map = {"RELIANCE.NS": 0.017, "TCS.NS": 0.013, "HDFCBANK.NS": 0.014, "INFY.NS": 0.015, "ICICIBANK.NS": 0.016, "LT.NS": 0.018, "SBIN.NS": 0.021, "ITC.NS": 0.011, "TITAN.NS": 0.017, "AXISBANK.NS": 0.019}
#     start_price = {"RELIANCE.NS": 2500, "TCS.NS": 3800, "HDFCBANK.NS": 1450, "INFY.NS": 1650, "ICICIBANK.NS": 1120, "LT.NS": 3400, "SBIN.NS": 780, "ITC.NS": 440, "TITAN.NS": 3300, "AXISBANK.NS": 1080}
#     drift = drift_map.get(symbol, 0.0004)
#     vol = vol_map.get(symbol, 0.018)
#     prices = [start_price.get(symbol, 100.0)]
#     returns = rng.normal(drift, vol, len(dates))
#     for r in returns[1:]:
#         prices.append(prices[-1] * np.exp(r))
#     close = np.array(prices)
#     open_ = close * (1 + rng.normal(0, 0.004, len(close)))
#     high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.006, 0.004, len(close))))
#     low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.006, 0.004, len(close))))
#     volume = np.clip(rng.lognormal(mean=14.2, sigma=0.35, size=len(close)), 250000, 25000000)
#     df = pd.DataFrame({"symbol": symbol, "date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
#     df["return_1d"] = df["close"].pct_change().fillna(0)
#     df["ma_10"] = df["close"].rolling(10).mean().bfill()
#     df["ma_20"] = df["close"].rolling(20).mean().bfill()
#     df["volatility_10"] = df["return_1d"].rolling(10).std().bfill().fillna(0)
#     df["momentum_20"] = df["close"].pct_change(20).bfill().fillna(0)
#     return df


# def build_multistock_dataset(universe):
#     frames = [generate_stock_history(sym) for sym in universe["symbol"]]
#     all_df = pd.concat(frames, ignore_index=True)
#     all_df.to_csv(DATA_PATH, index=False)
#     return all_df


# def create_sequences(arr, window=30):
#     X, y = [], []
#     for i in range(window, len(arr)):
#         X.append(arr[i-window:i])
#         y.append(arr[i, 0])
#     return np.array(X), np.array(y)


# def inverse_first_col_with_meta(scaled_first_col, meta):
#     data_min = np.array(meta["min_"])
#     data_max = np.array(meta["max_"])
#     scale = np.where((data_max - data_min) == 0, 1.0, data_max - data_min)
#     return np.array(scaled_first_col).reshape(-1) * scale[0] + data_min[0]


# def main():
#     import tensorflow as tf
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Concatenate, Input, RepeatVector, Flatten
#     from tensorflow.keras.callbacks import EarlyStopping

#     universe = create_stock_universe() if not UNIVERSE_PATH.exists() else pd.read_csv(UNIVERSE_PATH)
#     df = build_multistock_dataset(universe) if not DATA_PATH.exists() else pd.read_csv(DATA_PATH)
#     df["date"] = pd.to_datetime(df["date"])

#     feature_cols = ["open", "high", "low", "close", "volume", "return_1d", "ma_10", "ma_20", "volatility_10", "momentum_20"]
#     symbols = sorted(df["symbol"].unique().tolist())
#     symbol_to_id = {s: i for i, s in enumerate(symbols)}
#     df["symbol_id"] = df["symbol"].map(symbol_to_id)

#     all_X, all_y, all_sid = [], [], []
#     window = 30
#     train_scalers = {}

#     for sym in symbols:
#         sdf = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)
#         Xraw = sdf[feature_cols].copy().values.astype(float)
#         scaler = MinMaxScaler().fit(Xraw)
#         train_scalers[sym] = scaler
#         scaled = scaler.transform(Xraw)
#         X, y = create_sequences(scaled, window=window)
#         all_X.append(X)
#         all_y.append(y)
#         all_sid.extend([symbol_to_id[sym]] * len(X))

#     X = np.concatenate(all_X, axis=0)
#     y = np.concatenate(all_y, axis=0)
#     sid = np.array(all_sid)

#     idx = np.arange(len(X))
#     np.random.default_rng(42).shuffle(idx)
#     X, y, sid = X[idx], y[idx], sid[idx]
#     split = int(len(X) * 0.8)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]
#     sid_train, sid_test = sid[:split], sid[split:]

#     seq_len = X_train.shape[1]
#     n_features = X_train.shape[2]
#     seq_in = Input(shape=(seq_len, n_features), name="sequence")
#     sid_in = Input(shape=(1,), dtype="int32", name="symbol_id")
#     sid_emb = Embedding(input_dim=len(symbols), output_dim=4, name="symbol_embedding")(sid_in)
#     sid_emb = Flatten()(sid_emb)
#     sid_rep = RepeatVector(seq_len)(sid_emb)
#     x = Concatenate(axis=-1)([seq_in, sid_rep])
#     x = LSTM(64, return_sequences=True)(x)
#     x = Dropout(0.2)(x)
#     x = LSTM(32)(x)
#     x = Dropout(0.2)(x)
#     x = Dense(32, activation="relu")(x)
#     out = Dense(1)(x)
#     model = Model(inputs=[seq_in, sid_in], outputs=out)
#     model.compile(optimizer="adam", loss="mse")
#     es = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
#     model.fit([X_train, sid_train], y_train, validation_split=0.15, epochs=60, batch_size=32, callbacks=[es], verbose=0)

#     pred = model.predict([X_test, sid_test], verbose=0).reshape(-1)
#     true = y_test.reshape(-1)

#     test_symbol_ids = sid_test.reshape(-1)
#     pred_inv_all, true_inv_all = [], []
#     for sid_val in np.unique(test_symbol_ids):
#         sym = symbols[int(sid_val)]
#         mask = test_symbol_ids == sid_val
#         scaler = train_scalers[sym]
#         pred_inv_all.extend(list(inverse_first_col_with_meta(pred[mask], {
#             "min_": scaler.data_min_.tolist(),
#             "max_": scaler.data_max_.tolist(),
#             "scale_": scaler.scale_.tolist(),
#         })))
#         true_inv_all.extend(list(inverse_first_col_with_meta(true[mask], {
#             "min_": scaler.data_min_.tolist(),
#             "max_": scaler.data_max_.tolist(),
#             "scale_": scaler.scale_.tolist(),
#         })))

#     pred_inv = np.array(pred_inv_all)
#     true_inv = np.array(true_inv_all)
#     mse = mean_squared_error(true_inv, pred_inv)
#     metrics = {
#         "mae": float(mean_absolute_error(true_inv, pred_inv)),
#         "rmse": float(np.sqrt(mse)),
#         "mape": float(np.mean(np.abs((true_inv - pred_inv) / np.maximum(true_inv, 1e-6))) * 100),
#         "model_type": "TensorFlow Multi-Stock LSTM",
#         "symbols": symbols,
#     }

#     model.save(MODEL_PATH)
#     joblib.dump(train_scalers, SCALER_PATH)
#     json.dump(feature_cols, open(FEATURES_PATH, "w"), indent=2)
#     json.dump(metrics, open(METRICS_PATH, "w"), indent=2)
#     json.dump({"window": window, "symbols": symbols}, open(SCALER_META_PATH, "w"), indent=2)

#     future_rows = []
#     for sym in symbols:
#         sdf = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)
#         Xraw = sdf[feature_cols].copy().values.astype(float)
#         scaler = train_scalers[sym]
#         scaled = scaler.transform(Xraw)
#         current = scaled[-window:].copy()
#         s_id = np.array([[symbol_to_id[sym]]])
#         fut = []
#         for _ in range(30):
#             batch_x = np.expand_dims(current.astype(np.float32), axis=0)
#             batch_sid = np.array([[symbol_to_id[sym]]], dtype=np.int32)
#             next_pred = model.predict([batch_x, batch_sid], verbose=0)[0, 0]
#             fut.append(next_pred)
#             next_row = current[-1].copy()
#             next_row[3] = next_pred
#             current = np.vstack([current[1:], next_row])
#         future_close = inverse_first_col_with_meta(np.array(fut), {"min_": scaler.data_min_.tolist(), "max_": scaler.data_max_.tolist()})
#         future_dates = pd.bdate_range(sdf["date"].iloc[-1] + pd.offsets.BDay(1), periods=30)
#         future_rows.append(pd.DataFrame({"symbol": sym, "date": future_dates, "forecast_close": future_close}))

#     forecast_df = pd.concat(future_rows, ignore_index=True)
#     forecast_df.to_csv(FORECAST_PATH, index=False)
#     print(metrics)


# if __name__ == "__main__":
#     main()

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = BASE_DIR / "output"
OUT.mkdir(exist_ok=True)

UNIVERSE_PATH = OUT / "synthetic_stock_universe.csv"
DATA_PATH = OUT / "synthetic_stock_data.csv"
MODEL_PATH = OUT / "stock_lstm_model.keras"
SCALER_PATH = OUT / "stock_scaler.joblib"
FEATURES_PATH = OUT / "stock_feature_columns.json"
METRICS_PATH = OUT / "stock_metrics.json"
FORECAST_PATH = OUT / "stock_forecast_output.csv"
SCALER_META_PATH = OUT / "stock_scaler_meta.json"


def create_stock_universe():
    stocks = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "LT.NS",
        "SBIN.NS",
        "ITC.NS",
        "TITAN.NS",
        "AXISBANK.NS",
    ]
    sectors = [
        "Energy",
        "IT",
        "Banking",
        "IT",
        "Banking",
        "Construction",
        "Banking",
        "FMCG",
        "Consumer",
        "Banking",
    ]
    beta = [1.15, 0.95, 1.05, 0.90, 1.00, 1.10, 1.20, 0.85, 0.98, 1.08]
    rng = np.random.default_rng(7)

    df = pd.DataFrame(
        {
            "symbol": stocks,
            "sector": sectors,
            "beta": beta,
            "momentum": np.round(rng.uniform(0.2, 0.95, len(stocks)), 3),
            "valuation": np.round(rng.uniform(0.15, 0.85, len(stocks)), 3),
            "dividend_yield": np.round(rng.uniform(0.01, 0.035, len(stocks)), 4),
            "pe_ratio": np.round(rng.uniform(12, 45, len(stocks)), 2),
            "de_ratio": np.round(rng.uniform(0.05, 1.8, len(stocks)), 2),
        }
    )
    df.to_csv(UNIVERSE_PATH, index=False)
    return df


def generate_stock_history(symbol, n_days=1500, seed=42):
    seed_val = abs(hash(symbol)) % (2**32)
    rng = np.random.default_rng(seed + seed_val)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    drift_map = {
        "RELIANCE.NS": 0.0005,
        "TCS.NS": 0.00045,
        "HDFCBANK.NS": 0.00042,
        "INFY.NS": 0.00048,
        "ICICIBANK.NS": 0.00044,
        "LT.NS": 0.00041,
        "SBIN.NS": 0.00047,
        "ITC.NS": 0.00033,
        "TITAN.NS": 0.00046,
        "AXISBANK.NS": 0.00043,
    }
    vol_map = {
        "RELIANCE.NS": 0.017,
        "TCS.NS": 0.013,
        "HDFCBANK.NS": 0.014,
        "INFY.NS": 0.015,
        "ICICIBANK.NS": 0.016,
        "LT.NS": 0.018,
        "SBIN.NS": 0.021,
        "ITC.NS": 0.011,
        "TITAN.NS": 0.017,
        "AXISBANK.NS": 0.019,
    }
    start_price = {
        "RELIANCE.NS": 2500,
        "TCS.NS": 3800,
        "HDFCBANK.NS": 1450,
        "INFY.NS": 1650,
        "ICICIBANK.NS": 1120,
        "LT.NS": 3400,
        "SBIN.NS": 780,
        "ITC.NS": 440,
        "TITAN.NS": 3300,
        "AXISBANK.NS": 1080,
    }

    drift = drift_map.get(symbol, 0.0004)
    vol = vol_map.get(symbol, 0.018)

    prices = [start_price.get(symbol, 100.0)]
    returns = rng.normal(drift, vol, len(dates))
    for r in returns[1:]:
        prices.append(prices[-1] * np.exp(r))

    close = np.array(prices)
    open_ = close * (1 + rng.normal(0, 0.004, len(close)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.006, 0.004, len(close))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.006, 0.004, len(close))))
    volume = np.clip(rng.lognormal(mean=14.2, sigma=0.35, size=len(close)), 250000, 25000000)

    df = pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df["return_1d"] = df["close"].pct_change().fillna(0)
    df["ma_10"] = df["close"].rolling(10).mean().bfill()
    df["ma_20"] = df["close"].rolling(20).mean().bfill()
    df["volatility_10"] = df["return_1d"].rolling(10).std().bfill().fillna(0)
    df["momentum_20"] = df["close"].pct_change(20).bfill().fillna(0)
    return df


def build_multistock_dataset(universe):
    frames = [generate_stock_history(sym) for sym in universe["symbol"]]
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(DATA_PATH, index=False)
    return all_df


def create_sequences(arr, target_idx=3, window=30):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i])
        y.append(arr[i, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def inverse_target_with_scaler(scaled_vals, scaler, target_idx=3):
    scaled_vals = np.array(scaled_vals).reshape(-1)
    data_min = scaler.data_min_[target_idx]
    data_max = scaler.data_max_[target_idx]
    scale = data_max - data_min
    if scale == 0:
        return np.full_like(scaled_vals, fill_value=data_min, dtype=float)
    return scaled_vals * scale + data_min


def main():
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, Flatten, Input, LSTM, RepeatVector
    from tensorflow.keras.models import Model

    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    universe = create_stock_universe() if not UNIVERSE_PATH.exists() else pd.read_csv(UNIVERSE_PATH)
    df = build_multistock_dataset(universe) if not DATA_PATH.exists() else pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "ma_10",
        "ma_20",
        "volatility_10",
        "momentum_20",
    ]
    target_idx = feature_cols.index("close")

    symbols = sorted(df["symbol"].unique().tolist())
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    df["symbol_id"] = df["symbol"].map(symbol_to_id)

    all_X, all_y, all_sid = [], [], []
    window = 30
    train_scalers = {}

    for sym in symbols:
        sdf = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)
        Xraw = sdf[feature_cols].copy().values.astype(np.float32)

        scaler = MinMaxScaler().fit(Xraw)
        train_scalers[sym] = scaler

        scaled = scaler.transform(Xraw).astype(np.float32)
        X_sym, y_sym = create_sequences(scaled, target_idx=target_idx, window=window)

        all_X.append(X_sym)
        all_y.append(y_sym)
        all_sid.extend([symbol_to_id[sym]] * len(X_sym))

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    sid = np.array(all_sid, dtype=np.int32)

    idx = np.arange(len(X))
    np.random.default_rng(42).shuffle(idx)
    X, y, sid = X[idx], y[idx], sid[idx]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split].astype(np.float32), X[split:].astype(np.float32)
    y_train, y_test = y[:split].astype(np.float32), y[split:].astype(np.float32)
    sid_train = sid[:split].astype(np.int32).reshape(-1, 1)
    sid_test = sid[split:].astype(np.int32).reshape(-1, 1)

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    seq_in = Input(shape=(seq_len, n_features), name="sequence", dtype="float32")
    sid_in = Input(shape=(1,), dtype="int32", name="symbol_id")

    sid_emb = Embedding(input_dim=len(symbols), output_dim=4, name="symbol_embedding")(sid_in)
    sid_emb = Flatten()(sid_emb)
    sid_rep = RepeatVector(seq_len)(sid_emb)

    x = Concatenate(axis=-1)([seq_in, sid_rep])
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)

    model = Model(inputs={"sequence": seq_in, "symbol_id": sid_in}, outputs=out)
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

    model.fit(
        {"sequence": X_train, "symbol_id": sid_train},
        y_train,
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        callbacks=[es],
        verbose=0,
    )

    pred = model.predict({"sequence": X_test, "symbol_id": sid_test}, verbose=0).reshape(-1)
    true = y_test.reshape(-1)
    test_symbol_ids = sid_test.reshape(-1)

    pred_inv_all, true_inv_all = [], []
    for sid_val in np.unique(test_symbol_ids):
        sym = symbols[int(sid_val)]
        mask = test_symbol_ids == sid_val
        scaler = train_scalers[sym]

        pred_inv_all.extend(list(inverse_target_with_scaler(pred[mask], scaler, target_idx=target_idx)))
        true_inv_all.extend(list(inverse_target_with_scaler(true[mask], scaler, target_idx=target_idx)))

    pred_inv = np.array(pred_inv_all)
    true_inv = np.array(true_inv_all)

    mse = mean_squared_error(true_inv, pred_inv)
    metrics = {
        "mae": float(mean_absolute_error(true_inv, pred_inv)),
        "rmse": float(np.sqrt(mse)),
        "mape": float(np.mean(np.abs((true_inv - pred_inv) / np.maximum(true_inv, 1e-6))) * 100),
        "model_type": "TensorFlow Multi-Stock LSTM",
        "symbols": symbols,
    }

    model.save(MODEL_PATH)
    joblib.dump(train_scalers, SCALER_PATH)

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(SCALER_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"window": window, "symbols": symbols, "target_column": "close"}, f, indent=2)

    future_rows = []
    for sym in symbols:
        sdf = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)
        Xraw = sdf[feature_cols].copy().values.astype(np.float32)
        scaler = train_scalers[sym]
        scaled = scaler.transform(Xraw).astype(np.float32)

        current = scaled[-window:].copy()
        fut = []

        for _ in range(30):
            batch_x = np.expand_dims(current, axis=0).astype(np.float32)
            batch_sid = np.array([[symbol_to_id[sym]]], dtype=np.int32)

            next_pred = model(
                {"sequence": batch_x, "symbol_id": batch_sid},
                training=False
            ).numpy()[0, 0]

            fut.append(float(next_pred))

            next_row = current[-1].copy()
            next_row[target_idx] = next_pred
            current = np.vstack([current[1:], next_row]).astype(np.float32)

        future_close = inverse_target_with_scaler(np.array(fut), scaler, target_idx=target_idx)
        future_dates = pd.bdate_range(sdf["date"].iloc[-1] + pd.offsets.BDay(1), periods=30)

        future_rows.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": future_dates,
                    "forecast_close": future_close,
                }
            )
        )

    forecast_df = pd.concat(future_rows, ignore_index=True)
    forecast_df.to_csv(FORECAST_PATH, index=False)

    print(metrics)


if __name__ == "__main__":
    main()