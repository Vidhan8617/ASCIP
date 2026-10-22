
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="ASCIP | Supply Chain Intelligence",
    page_icon="📦",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
[data-testid="stMetric"] {
    background: #172033;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #28354d;
}
</style>
""", unsafe_allow_html=True)

st.title("📦 ASCIP")
st.caption("AI-Powered Supply Chain Intelligence Platform")

if st.button("🔄 Refresh dashboard"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data(ttl=30)
def fetch_data(endpoint):
    response = requests.get(f"{API_URL}/{endpoint}", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_ai_insights(endpoint):
    response = requests.get(f"{API_URL}/{endpoint}", timeout=180)
    response.raise_for_status()
    return response.json()


try:
    health = fetch_data("health")
    summary = fetch_data("summary")
    suppliers = pd.DataFrame(fetch_data("suppliers/risks"))
    inventory = pd.DataFrame(fetch_data("inventory/alerts"))
except requests.RequestException as e:
    st.error("Cannot connect to the ASCIP API or fetch its data.")
    st.caption(f"Details: {e}")
    st.code("python -m uvicorn api.main:app --reload")
    st.stop()

st.success("ASCIP API connected successfully")

st.subheader("Supply Chain Overview")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records", f"{summary['total_records']:,}")
c2.metric("Products", summary["total_products"])
c3.metric("Suppliers", summary["total_suppliers"])
c4.metric("High-Risk Suppliers", summary["high_risk_suppliers"])
c5.metric("High-Risk Inventory", summary["high_risk_inventory_items"])

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Supplier Risk Analysis")
    if not suppliers.empty and "risk_level" in suppliers.columns:
        risk_counts = (
            suppliers["risk_level"]
            .value_counts()
            .rename_axis("Risk Level")
            .reset_index(name="Suppliers")
        )
        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Suppliers",
            hole=0.55,
            title="Supplier Risk Distribution",
            color="Risk Level",
            color_discrete_map={
                "Low": "#22c55e",
                "Medium": "#f59e0b",
                "High": "#ef4444"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No supplier risk data available.")

with right:
    st.subheader("Inventory Risk Analysis")
    if not inventory.empty and "stockout_risk" in inventory.columns:
        inventory_counts = (
            inventory["stockout_risk"]
            .value_counts()
            .rename_axis("Risk Level")
            .reset_index(name="Items")
        )
        fig = px.bar(
            inventory_counts,
            x="Risk Level",
            y="Items",
            color="Risk Level",
            title="Inventory Items by Risk Level",
            color_discrete_map={
                "Low": "#22c55e",
                "Medium": "#f59e0b",
                "High": "#ef4444"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No inventory risk data available.")

st.divider()

st.subheader("Supplier Risk Rankings")

if not suppliers.empty and "risk_level" in suppliers.columns:
    risk_options = [
        level for level in ["High", "Medium", "Low"]
        if level in suppliers["risk_level"].astype(str).unique()
    ]
    risk_filter = st.multiselect(
        "Filter by supplier risk",
        options=risk_options,
        default=risk_options,
        key="supplier_filter"
    )
    filtered_suppliers = suppliers[
        suppliers["risk_level"].isin(risk_filter)
    ].copy()

    if "risk_score" in filtered_suppliers.columns:
        filtered_suppliers = filtered_suppliers.sort_values(
            "risk_score", ascending=False
        )

    st.dataframe(
        filtered_suppliers,
        use_container_width=True,
        hide_index=True
    )
    st.download_button(
        "Download Supplier Report",
        data=filtered_suppliers.to_csv(index=False),
        file_name="ascip_supplier_report.csv",
        mime="text/csv"
    )
else:
    st.info("No supplier data available.")

st.subheader("Inventory Alerts")

if not inventory.empty and "stockout_risk" in inventory.columns:
    inventory_options = [
        level for level in ["High", "Medium", "Low"]
        if level in inventory["stockout_risk"].astype(str).unique()
    ]
    inventory_filter = st.multiselect(
        "Filter by stockout risk",
        options=inventory_options,
        default=inventory_options,
        key="inventory_filter"
    )
    filtered_inventory = inventory[
        inventory["stockout_risk"].isin(inventory_filter)
    ].copy()

    if "suggested_reorder_qty" in filtered_inventory.columns:
        filtered_inventory = filtered_inventory.sort_values(
            "suggested_reorder_qty", ascending=False
        )

    st.dataframe(
        filtered_inventory,
        use_container_width=True,
        hide_index=True
    )
    st.download_button(
        "Download Inventory Report",
        data=filtered_inventory.to_csv(index=False),
        file_name="ascip_inventory_report.csv",
        mime="text/csv"
    )
else:
    st.info("No inventory data available.")

st.divider()
st.subheader("Gemini AI Insights")
st.caption(
    "Generate explanations and recommended actions on demand. "
    "Up to 20 priority items per analysis are sent to Gemini."
)

if st.button("🧠 Generate Supplier AI Insights"):
    with st.spinner("Analyzing supplier risks..."):
        try:
            data = fetch_ai_insights("insights/suppliers")
            result = pd.DataFrame(data)

            if result.empty:
                st.info("No supplier insights available.")
            else:
                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button(
                    "Download Supplier AI Insights",
                    data=result.to_csv(index=False),
                    file_name="ascip_supplier_ai_insights.csv",
                    mime="text/csv",
                    key="download_supplier_ai"
                )
        except requests.RequestException as e:
            st.error(f"Could not generate supplier insights: {e}")

if st.button("📦 Generate Inventory AI Insights"):
    with st.spinner("Analyzing inventory risks..."):
        try:
            data = fetch_ai_insights("insights/inventory")
            result = pd.DataFrame(data)

            if result.empty:
                st.info("No inventory insights available.")
            else:
                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button(
                    "Download Inventory AI Insights",
                    data=result.to_csv(index=False),
                    file_name="ascip_inventory_ai_insights.csv",
                    mime="text/csv",
                    key="download_inventory_ai"
                )
        except requests.RequestException as e:
            st.error(f"Could not generate inventory insights: {e}")

st.caption(
    "ASCIP MVP | Synthetic supply-chain data | "
    "AI-assisted decision support, not live operational data"
)
