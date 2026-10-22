
import pandas as pd

def calculate_supplier_risk(df):
    suppliers = df.groupby("supplier_id").agg(
        avg_delivery_days=("delivery_days", "mean"),
        avg_defect_rate=("defect_rate", "mean"),
        on_time_rate=("supplier_on_time", "mean"),
        total_orders=("order_quantity", lambda x: (x > 0).sum()),
        stockouts=("stockout", "sum")
    ).reset_index()

    delivery_score = (suppliers["avg_delivery_days"] / max(suppliers["avg_delivery_days"].max(), 1) * 100).clip(0, 100)
    defect_score = (suppliers["avg_defect_rate"] / max(suppliers["avg_defect_rate"].max(), 0.0001) * 100).clip(0, 100)
    reliability_score = (1 - suppliers["on_time_rate"]) * 100

    suppliers["risk_score"] = (
        delivery_score * 0.30 +
        defect_score * 0.35 +
        reliability_score * 0.35
    ).round(2)

    suppliers["risk_level"] = pd.cut(
        suppliers["risk_score"],
        bins=[-1, 35, 65, 100],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    return suppliers.sort_values("risk_score", ascending=False)
