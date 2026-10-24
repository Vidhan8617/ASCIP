
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "supply_chain_data.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna(subset=["date", "demand", "product_id"])
    df["warehouse"] = df["warehouse"].fillna("Unknown").astype(str)
    df["product_id"] = df["product_id"].astype(str)
    return df

def prepare_features(df):
    df = df.groupby(
        ["date", "product_id", "warehouse"], as_index=False
    ).agg(demand=("demand", "sum"))

    df = df.sort_values(["product_id", "warehouse", "date"])
    groups = ["product_id", "warehouse"]
    grouped = df.groupby(groups)["demand"]

    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["rolling_mean_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).mean()
    )
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["product_code"] = pd.factorize(df["product_id"])[0]
    df["warehouse_code"] = pd.factorize(df["warehouse"])[0]

    return df.dropna().copy()

def forecast_demand(horizon=14):
    if not 1 <= horizon <= 90:
        raise ValueError("Horizon must be between 1 and 90 days.")

    raw = load_data()
    df = prepare_features(raw)

    if len(df) < 30:
        raise ValueError("Not enough historical data to train the model.")

    dates = np.sort(df["date"].unique())
    split_index = int(len(dates) * 0.8)

    if split_index == 0 or split_index >= len(dates):
        raise ValueError("Not enough dates for chronological evaluation.")

    split_date = dates[split_index]
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]

    features = [
        "lag_1", "lag_7", "rolling_mean_7",
        "day_of_week", "month", "product_code", "warehouse_code"
    ]

    if train.empty or test.empty:
        raise ValueError("Unable to create training and test sets.")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train[features], train["demand"])

    test_predictions = np.maximum(
        model.predict(test[features]), 0
    )

    metrics = {
        "mae": round(float(mean_absolute_error(
            test["demand"], test_predictions
        )), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(
            test["demand"], test_predictions
        ))), 3),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "evaluation_start": str(pd.Timestamp(split_date).date())
    }

    history = raw.groupby(
        ["product_id", "warehouse", "date"], as_index=False
    )["demand"].sum().sort_values(["product_id", "warehouse", "date"])

    forecasts = []

    for (product, warehouse), group in history.groupby(
        ["product_id", "warehouse"]
    ):
        group = group.sort_values("date")
        values = group["demand"].astype(float).tolist()

        if len(values) < 7:
            continue

        product_code = int(
            df.loc[df["product_id"] == product, "product_code"].iloc[0]
        )
        warehouse_code = int(
            df.loc[
                (df["product_id"] == product) &
                (df["warehouse"] == warehouse),
                "warehouse_code"
            ].iloc[0]
        )

        last_date = group["date"].max()

        for step in range(1, horizon + 1):
            future_date = last_date + pd.Timedelta(days=step)

            row = pd.DataFrame([{
                "lag_1": values[-1],
                "lag_7": values[-7],
                "rolling_mean_7": float(np.mean(values[-7:])),
                "day_of_week": future_date.dayofweek,
                "month": future_date.month,
                "product_code": product_code,
                "warehouse_code": warehouse_code
            }])

            prediction = max(0.0, float(model.predict(row[features])[0]))
            values.append(prediction)

            forecasts.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "product_id": product,
                "warehouse": warehouse,
                "predicted_demand": round(prediction, 2)
            })

    forecast_df = pd.DataFrame(forecasts)

    return {
        "metrics": metrics,
        "forecast": forecast_df.to_dict(orient="records")
    }
