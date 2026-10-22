
from pathlib import Path
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ai_insights import generate_supplier_insights, generate_inventory_insights
from supplier_risk import calculate_supplier_risk
from inventory_intelligence import analyze_inventory

app = FastAPI(
    title="ASCIP API",
    description="AI-powered Supply Chain Intelligence Platform",
    version="1.0.0"
)

DATA_PATH = ROOT / "data" / "raw" / "supply_chain_data.csv"
df = None

@app.on_event("startup")
def load_data():
    global df
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        raise RuntimeError(f"Dataset not found: {DATA_PATH}")

@app.get("/")
def home():
    return {
        "project": "ASCIP",
        "status": "running",
        "message": "Supply Chain Intelligence API"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/suppliers/risks")
def supplier_risks():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    return calculate_supplier_risk(df).to_dict(orient="records")

@app.get("/inventory/alerts")
def inventory_alerts():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    return analyze_inventory(df).to_dict(orient="records")

@app.get("/summary")
def summary():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    suppliers = calculate_supplier_risk(df)
    inventory = analyze_inventory(df)

    return {
        "total_records": len(df),
        "total_products": int(df["product_id"].nunique()),
        "total_suppliers": int(df["supplier_id"].nunique()),
        "high_risk_suppliers": int((suppliers["risk_level"] == "High").sum()),
        "high_risk_inventory_items": int((inventory["stockout_risk"] == "High").sum())
    }

@app.get("/insights/suppliers")
def supplier_insights():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    results = generate_supplier_insights(calculate_supplier_risk(df))
    return results.to_dict(orient="records")

@app.get("/insights/inventory")
def inventory_insights():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    results = generate_inventory_insights(analyze_inventory(df))
    return results.to_dict(orient="records")
