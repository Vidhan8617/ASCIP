from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from supplier_risk import calculate_supplier_risk
from inventory_intelligence import analyze_inventory

df = pd.read_csv(ROOT / "data" / "raw" / "supply_chain_data.csv")

suppliers = calculate_supplier_risk(df)
inventory = analyze_inventory(df)

print("\nSUPPLIER RISK COLUMNS:")
print(suppliers.columns.tolist())
print(suppliers.head(10).to_string(index=False))

print("\nINVENTORY ANALYSIS COLUMNS:")
print(inventory.columns.tolist())
print(inventory.head(10).to_string(index=False))