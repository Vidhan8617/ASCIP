from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "supply_chain_data.csv"
FEATURES = ["lag_1", "lag_7", "rolling_mean_7", "day_of_week", "month", "product_code", "warehouse_code"]
BACKTEST_HORIZONS = (7, 14, 30)
BACKTEST_ORIGINS = 1


def load_data():
    df = pd.read_csv(DATA_PATH)
    required = {"date", "demand", "product_id", "warehouse"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna(subset=["date", "demand", "product_id"])
    df["warehouse"] = df["warehouse"].fillna("Unknown").astype(str)
    df["product_id"] = df["product_id"].astype(str)
    return df.groupby(["date", "product_id", "warehouse"], as_index=False).agg(demand=("demand", "sum")).sort_values(["product_id", "warehouse", "date"]).reset_index(drop=True)


def prepare_features(df):
    df = df.copy().sort_values(["product_id", "warehouse", "date"]).reset_index(drop=True)
    grouped = df.groupby(["product_id", "warehouse"])["demand"]
    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["rolling_mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    product_codes = {v: i for i, v in enumerate(sorted(df["product_id"].unique()))}
    warehouse_codes = {v: i for i, v in enumerate(sorted(df["warehouse"].unique()))}
    df["product_code"] = df["product_id"].map(product_codes).astype(int)
    df["warehouse_code"] = df["warehouse"].map(warehouse_codes).astype(int)
    return df.dropna(subset=FEATURES + ["demand"]).copy()


def create_model():
    return RandomForestRegressor(n_estimators=30, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)


def calculate_metrics(actual, predicted):
    if len(actual) == 0:
        return {"mae": None, "rmse": None, "rows": 0}
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 3),
        "rows": int(len(actual))
    }


def recursive_batch_predictions(model, histories, product_codes, warehouse_codes, max_horizon):
    """Forecast all product-warehouse series together, making one predict call per future day."""
    states = {}
    for key, group in histories.items():
        group = group.sort_values("date")
        values = group["demand"].astype(float).tolist()
        if len(values) >= 7:
            states[key] = {"values": values, "last_date": pd.Timestamp(group["date"].max()), "predictions": []}

    for step in range(1, max_horizon + 1):
        rows = []
        keys = []
        for (product, warehouse), state in states.items():
            values = state["values"]
            future_date = state["last_date"] + pd.Timedelta(days=step)
            rows.append({
                "lag_1": values[-1],
                "lag_7": values[-7],
                "rolling_mean_7": float(np.mean(values[-7:])),
                "day_of_week": future_date.dayofweek,
                "month": future_date.month,
                "product_code": product_codes[product],
                "warehouse_code": warehouse_codes[warehouse]
            })
            keys.append((product, warehouse))
        if not rows:
            break
        batch = pd.DataFrame(rows, columns=FEATURES)
        batch_predictions = np.maximum(model.predict(batch), 0.0)
        for key, prediction in zip(keys, batch_predictions):
            value = float(prediction)
            states[key]["predictions"].append(value)
            states[key]["values"].append(value)
    return states


def run_backtest(raw, product_codes, warehouse_codes):
    dates = np.sort(raw["date"].unique())
    max_horizon = max(BACKTEST_HORIZONS)
    first_origin_index = max(30, int(len(dates) * 0.65))
    possible_indices = list(range(first_origin_index, len(dates) - max_horizon + 1))
    if not possible_indices:
        raise ValueError("Not enough dates to run a 30-day backtest.")
    count = min(BACKTEST_ORIGINS, len(possible_indices))
    selected_positions = np.linspace(0, len(possible_indices) - 1, count, dtype=int)
    origin_indices = sorted({possible_indices[i] for i in selected_positions})
    rows = []

    for origin_index in origin_indices:
        origin_date = pd.Timestamp(dates[origin_index])
        train_raw = raw[raw["date"] < origin_date]
        train_features = prepare_features(train_raw)
        if len(train_features) < 30:
            continue
        model = create_model()
        model.fit(train_features[FEATURES], train_features["demand"])
        actual_future = raw[(raw["date"] >= origin_date) & (raw["date"] < origin_date + pd.Timedelta(days=max_horizon))]
        actual_lookup = {(r.product_id, r.warehouse, pd.Timestamp(r.date)): float(r.demand) for r in actual_future.itertuples(index=False)}
        histories = {key: group for key, group in train_raw.groupby(["product_id", "warehouse"])}
        states = recursive_batch_predictions(model, histories, product_codes, warehouse_codes, max_horizon)

        for (product, warehouse), state in states.items():
            group = histories[(product, warehouse)].sort_values("date")
            previous_value = group["demand"].astype(float).iloc[-1]
            seasonal_values = group["demand"].astype(float).tolist()
            last_date = pd.Timestamp(group["date"].max())
            for step, rf_prediction in enumerate(state["predictions"], start=1):
                target_date = last_date + pd.Timedelta(days=step)
                actual = actual_lookup.get((product, warehouse, target_date))
                previous_prediction = max(0.0, float(previous_value))
                seasonal_prediction = max(0.0, float(seasonal_values[-7]))
                if actual is not None:
                    rows.append({
                        "origin": origin_date.strftime("%Y-%m-%d"), "date": target_date,
                        "product_id": product, "warehouse": warehouse, "horizon_day": step,
                        "actual": actual, "previous_day": previous_prediction,
                        "seasonal_7_day": seasonal_prediction, "random_forest": rf_prediction
                    })
                seasonal_values.append(seasonal_prediction)

    if not rows:
        raise ValueError("Backtest produced no comparable actual demand rows.")
    result = pd.DataFrame(rows)
    comparisons = {}
    for horizon in BACKTEST_HORIZONS:
        subset = result[result["horizon_day"] <= horizon]
        comparisons[str(horizon)] = {
            "days": horizon,
            "previous_day": calculate_metrics(subset["actual"], subset["previous_day"]),
            "seasonal_7_day": calculate_metrics(subset["actual"], subset["seasonal_7_day"]),
            "random_forest": calculate_metrics(subset["actual"], subset["random_forest"]),
            "origins": sorted(subset["origin"].unique())
        }
    return comparisons


def forecast_demand(horizon=14):
    if not 1 <= horizon <= 90:
        raise ValueError("Horizon must be between 1 and 90 days.")
    raw = load_data()
    if raw.empty:
        raise ValueError("The dataset contains no usable demand records.")
    product_codes = {v: i for i, v in enumerate(sorted(raw["product_id"].unique()))}
    warehouse_codes = {v: i for i, v in enumerate(sorted(raw["warehouse"].unique()))}
    df = prepare_features(raw)
    if len(df) < 30:
        raise ValueError("Not enough historical data to train the model.")
    dates = np.sort(df["date"].unique())
    split_index = int(len(dates) * 0.8)
    if split_index == 0 or split_index >= len(dates):
        raise ValueError("Not enough dates for chronological evaluation.")
    split_date = pd.Timestamp(dates[split_index])
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    if train.empty or test.empty:
        raise ValueError("Unable to create training and test sets.")
    model = create_model()
    model.fit(train[FEATURES], train["demand"])
    test_predictions = np.maximum(model.predict(test[FEATURES]), 0)
    metrics = calculate_metrics(test["demand"], test_predictions)
    metrics.update({
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "evaluation_start": split_date.date().isoformat(),
        "evaluation_type": "rolling one-step-ahead with observed lag features"
    })

    histories = {key: group for key, group in raw.groupby(["product_id", "warehouse"])}
    states = recursive_batch_predictions(model, histories, product_codes, warehouse_codes, horizon)
    forecasts = []
    for (product, warehouse), state in states.items():
        last_date = state["last_date"]
        for step, prediction in enumerate(state["predictions"], start=1):
            forecasts.append({
                "date": (last_date + pd.Timedelta(days=step)).strftime("%Y-%m-%d"),
                "product_id": product, "warehouse": warehouse,
                "predicted_demand": round(float(prediction), 2)
            })
    backtest = run_backtest(raw, product_codes, warehouse_codes)
    return {"metrics": metrics, "backtest": backtest, "forecast": forecasts}
