
import numpy as np
import pandas as pd

def analyze_inventory(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    rows = []

    for (product_id, warehouse), group in df.groupby(["product_id", "warehouse"]):
        group = group.sort_values("date")
        recent = group.tail(14)

        latest = group.iloc[-1]
        avg_demand = recent["demand"].mean()
        demand_std = recent["demand"].std()
        demand_std = 0 if pd.isna(demand_std) else demand_std
        lead_time = max(1, int(round(group["delivery_days"].mean())))
        inventory = max(0, latest["inventory"])

        safety_stock = 1.65 * demand_std * np.sqrt(lead_time)
        reorder_point = avg_demand * lead_time + safety_stock
        target_stock = avg_demand * (lead_time + 7) + safety_stock
        reorder_qty = max(0, int(np.ceil(target_stock - inventory)))

        stockout_risk = (
            "High" if inventory <= reorder_point * 0.5
            else "Medium" if inventory <= reorder_point
            else "Low"
        )

        rows.append({
            "product_id": product_id,
            "product_name": latest["product_name"],
            "warehouse": warehouse,
            "current_inventory": inventory,
            "avg_daily_demand": round(avg_demand, 2),
            "estimated_lead_time_days": lead_time,
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2),
            "suggested_reorder_qty": reorder_qty,
            "stockout_risk": stockout_risk
        })

    return pd.DataFrame(rows).sort_values(
        ["stockout_risk", "suggested_reorder_qty"],
        ascending=[True, False],
        key=lambda col: col.map({"High": 0, "Medium": 1, "Low": 2}) if col.name == "stockout_risk" else col
    )
