
from pathlib import Path
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC)) if str(SRC) not in sys.path else None

from ai_insights import generate_supplier_insights, generate_inventory_insights
from supplier_risk import calculate_supplier_risk
from inventory_intelligence import analyze_inventory
from rag_chatbot import answer_question
from demand_forecasting import forecast_demand

app = FastAPI(
    title="ASCIP API",
    description="AI-powered Supply Chain Intelligence Platform",
    version="1.2.0"
)

DATA_PATH = ROOT / "data" / "raw" / "supply_chain_data.csv"
df = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list, max_length=20)


@app.on_event("startup")
def load_data():
    global df
    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)


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


@app.get("/forecast/demand")
def demand_forecast(horizon: int = Query(default=14, ge=1, le=90)):
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    try:
        return forecast_demand(horizon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.post("/chat")
def chat(request: ChatRequest):
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = answer_question(question, request.history)

    if result["status"] == "error":
        raise HTTPException(status_code=502, detail=result["answer"])

    if result["status"] == "missing_api_key":
        raise HTTPException(status_code=503, detail=result["answer"])

    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "status": result["status"]
    }
