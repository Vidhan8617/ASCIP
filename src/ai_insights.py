
import os
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None
MODEL = "gemini-3.5-flash-lite"
MAX_AI_ITEMS = 20


def generate_ai_response(prompt):
    if client is None:
        return None
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        result = json.loads(response.text)
        return result if isinstance(result, dict) else None
    except Exception as e:
        print(f"Gemini request failed: {e}")
        return None


def supplier_fallback(row):
    reasons = []
    actions = []

    if float(row.get("avg_delivery_days", 0)) >= 8:
        reasons.append("long average delivery time")
        actions.append("Review delivery schedules and lead times")

    if float(row.get("avg_defect_rate", 0)) >= 0.04:
        reasons.append("high defect rate")
        actions.append("Conduct a supplier quality review")

    if float(row.get("on_time_rate", 0)) < 0.90:
        reasons.append("low on-time delivery performance")
        actions.append("Agree on delivery performance targets")

    if int(row.get("stockouts", 0)) >= 75:
        reasons.append("frequent stockouts")
        actions.append("Review safety stock and replenishment plans")

    if not reasons:
        reasons.append("no major threshold breaches detected")
        actions.append("Continue routine performance monitoring")

    return {
        "supplier_id": str(row.get("supplier_id", "Unknown")),
        "risk_level": str(row.get("risk_level", "Unknown")),
        "risk_score": float(row.get("risk_score", 0)),
        "risk_explanation": ", ".join(reasons),
        "recommended_actions": actions,
        "insight_source": "Rule-based fallback"
    }


def inventory_fallback(row):
    current = float(row.get("current_inventory", 0))
    reorder_point = float(row.get("reorder_point", 0))
    risk = str(row.get("stockout_risk", "Unknown"))

    explanation = (
        "Current inventory is at or below the reorder point"
        if current <= reorder_point
        else "Inventory is above the reorder point; continue monitoring"
    )

    if risk == "High":
        action = "Prioritize replenishment and verify supplier lead time"
    elif risk == "Medium":
        action = "Review upcoming demand and prepare a replenishment plan"
    else:
        action = "Monitor stock levels and follow the regular replenishment plan"

    return {
        "product_id": str(row.get("product_id", "Unknown")),
        "product_name": str(row.get("product_name", row.get("product_id", "Unknown"))),
        "warehouse": str(row.get("warehouse", "Unknown")),
        "stockout_risk": risk,
        "risk_explanation": explanation,
        "recommended_reorder_qty": max(
            0, int(round(float(row.get("suggested_reorder_qty", 0))))
        ),
        "recommended_action": action,
        "insight_source": "Rule-based fallback"
    }


def generate_supplier_insights(suppliers):
    if suppliers.empty:
        return pd.DataFrame()

    rows = suppliers.to_dict(orient="records")
    results = [supplier_fallback(row) for row in rows]

    priority = suppliers.copy()
    if "risk_score" in priority.columns:
        priority = priority.sort_values("risk_score", ascending=False)

    selected = priority.head(MAX_AI_ITEMS)
    selected_ids = [str(row.get("supplier_id", "Unknown")) for row in selected.to_dict(orient="records")]

    prompt_data = [
        {
            key: value
            for key, value in row.items()
            if not pd.isna(value) if not isinstance(value, (list, dict))
        }
        for row in selected.to_dict(orient="records")
    ]

    prompt = f"""
You are a supply chain risk analyst.
Analyze the supplier data provided. Use only supplied metrics and do not invent facts.
Return JSON with one key called "insights", containing a list of objects.
Each object must contain "supplier_id", "risk_explanation", and "recommended_actions".
"risk_explanation" must be a concise evidence-based string.
"recommended_actions" must be a list of 2 practical actions.
Include one object for each supplier provided.
Supplier data:
{json.dumps(prompt_data, default=str)}
"""

    ai_result = generate_ai_response(prompt)
    if ai_result and isinstance(ai_result.get("insights"), list):
        ai_by_id = {
            str(item.get("supplier_id")): item
            for item in ai_result["insights"]
            if isinstance(item, dict) and item.get("supplier_id") is not None
        }

        for index, row in enumerate(rows):
            supplier_id = str(row.get("supplier_id", "Unknown"))
            if supplier_id not in selected_ids or supplier_id not in ai_by_id:
                continue

            item = ai_by_id[supplier_id]
            explanation = item.get("risk_explanation")
            actions = item.get("recommended_actions")

            if isinstance(explanation, str) and explanation.strip() and isinstance(actions, list) and actions:
                results[index]["risk_explanation"] = explanation.strip()
                results[index]["recommended_actions"] = [
                    str(action) for action in actions[:3]
                ]
                results[index]["insight_source"] = "Gemini"

    return pd.DataFrame(results)


def generate_inventory_insights(inventory):
    if inventory.empty:
        return pd.DataFrame()

    rows = inventory.to_dict(orient="records")
    results = [inventory_fallback(row) for row in rows]

    priority = inventory.copy()
    if "stockout_risk" in priority.columns:
        priority["_priority"] = priority["stockout_risk"].map(
            {"High": 3, "Medium": 2, "Low": 1}
        ).fillna(0)
        sort_columns = ["_priority"]
        ascending = [False]

        if "suggested_reorder_qty" in priority.columns:
            sort_columns.append("suggested_reorder_qty")
            ascending.append(False)

        priority = priority.sort_values(sort_columns, ascending=ascending)

    selected = priority.head(MAX_AI_ITEMS)
    selected_ids = [str(row.get("product_id", "Unknown")) for row in selected.to_dict(orient="records")]

    
    prompt_data = [
        {
            key: value
            for key, value in row.items()
            if not isinstance(value, (list, dict)) and not pd.isna(value)
        }
        for row in selected.to_dict(orient="records")
    ]


    prompt = f"""
You are a supply chain inventory analyst.
Analyze the inventory data provided. Use only supplied metrics and do not invent facts or quantities.
Return JSON with one key called "insights", containing a list of objects.
Each object must contain "product_id", "risk_explanation", and "recommended_action".
Both explanation and action must be concise strings.
Include one object for each product provided.
Inventory data:
{json.dumps(prompt_data, default=str)}
"""

    ai_result = generate_ai_response(prompt)
    if ai_result and isinstance(ai_result.get("insights"), list):
        ai_by_id = {
            str(item.get("product_id")): item
            for item in ai_result["insights"]
            if isinstance(item, dict) and item.get("product_id") is not None
        }

        for index, row in enumerate(rows):
            product_id = str(row.get("product_id", "Unknown"))
            if product_id not in selected_ids or product_id not in ai_by_id:
                continue

            item = ai_by_id[product_id]
            explanation = item.get("risk_explanation")
            action = item.get("recommended_action")

            if isinstance(explanation, str) and explanation.strip() and isinstance(action, str) and action.strip():
                results[index]["risk_explanation"] = explanation.strip()
                results[index]["recommended_action"] = action.strip()
                results[index]["insight_source"] = "Gemini"

    return pd.DataFrame(results)
