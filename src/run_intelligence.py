
from pathlib import Path
import pandas as pd

from supplier_risk import calculate_supplier_risk
from inventory_intelligence import analyze_inventory

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "supply_chain_data.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

supplier_results = calculate_supplier_risk(df)
inventory_results = analyze_inventory(df)

supplier_results.to_csv(OUTPUT_DIR / "supplier_risk.csv", index=False)
inventory_results.to_csv(OUTPUT_DIR / "inventory_intelligence.csv", index=False)

print("\nTOP SUPPLIER RISKS")
print(supplier_results.head(10).to_string(index=False))

print("\nTOP INVENTORY ALERTS")
print(inventory_results.head(15).to_string(index=False))

print("\nResults saved to:", OUTPUT_DIR)
