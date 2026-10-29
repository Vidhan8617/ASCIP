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
.chat-answer {
    padding: 16px;
    border-radius: 12px;
    background: #172033;
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


@st.cache_data(ttl=300)
def fetch_forecast(horizon):
    response = requests.get(
        f"{API_URL}/forecast/demand",
        params={"horizon": horizon},
        timeout=180
    )
    response.raise_for_status()
    return response.json()


def ask_ascip(question, history):
    response = requests.post(
        f"{API_URL}/chat",
        json={"question": question, "history": history},
        timeout=180
    )
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

st.subheader("📈 Demand Forecasting")
st.caption(
    "Random Forest demand forecasts by product and warehouse. "
    "Forecasts are estimates based on historical demand."
)

forecast_col, button_col = st.columns([1, 2])
with forecast_col:
    forecast_horizon = st.selectbox(
        "Forecast horizon",
        options=[7, 14, 30, 60],
        index=1,
        format_func=lambda value: f"{value} days",
        key="forecast_horizon"
    )
with button_col:
    st.write("")
    st.write("")
    run_forecast = st.button(
        "📈 Generate Demand Forecast",
        use_container_width=True
    )

if run_forecast:
    with st.spinner("Training the forecasting model and generating predictions..."):
        try:
            st.session_state.ascip_forecast_result = fetch_forecast(forecast_horizon)
            st.session_state.ascip_forecast_horizon = forecast_horizon
            st.session_state.pop("ascip_forecast_error", None)
        except requests.RequestException as e:
            st.session_state.ascip_forecast_error = str(e)

if st.session_state.get("ascip_forecast_error"):
    st.error(f"Could not generate demand forecast: {st.session_state.ascip_forecast_error}")
    st.caption("Confirm that FastAPI is running and /forecast/demand is available.")

forecast_result = st.session_state.get("ascip_forecast_result")

if forecast_result:
    metrics = forecast_result.get("metrics", {})
    forecast_df = pd.DataFrame(forecast_result.get("forecast", []))

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", metrics.get("mae", "N/A"))
        m2.metric("RMSE", metrics.get("rmse", "N/A"))
        m3.metric("Training Rows", f"{metrics.get('train_rows', 0):,}")
        m4.metric("Test Rows", f"{metrics.get('test_rows', 0):,}")
        if metrics.get("evaluation_start"):
            st.caption(f"Chronological evaluation starts on {metrics['evaluation_start']}.")

    backtest = forecast_result.get("backtest", {})
    if backtest:
        st.divider()
        st.subheader("Forecast Evaluation")
        st.caption(
            "Historical backtest comparison. Scores are pooled across forecast days "
            "up to each horizon. The current run uses the forecast origins listed below; "
            "a single origin is preliminary evidence, not proof of general performance."
        )

        model_labels = {
            "random_forest": "Random Forest",
            "previous_day": "Previous-day baseline",
            "seasonal_7_day": "7-day seasonal baseline"
        }
        metric_rows = []
        origins = set()
        for horizon_key, comparison in sorted(backtest.items(), key=lambda item: int(item[0])):
            origins.update(comparison.get("origins", []))
            for model_key, model_label in model_labels.items():
                scores = comparison.get(model_key, {})
                metric_rows.append({
                    "Horizon (days)": int(horizon_key),
                    "Model": model_label,
                    "MAE": scores.get("mae"),
                    "RMSE": scores.get("rmse"),
                    "Rows": scores.get("rows", 0)
                })

        if metric_rows:
            evaluation_df = pd.DataFrame(metric_rows)
            st.caption(
                "Backtest origin(s): " + (", ".join(sorted(origins)) if origins else "not supplied")
            )
            st.dataframe(
                evaluation_df.sort_values(["Horizon (days)", "MAE"]),
                use_container_width=True,
                hide_index=True
            )

            mae_fig = px.bar(
                evaluation_df,
                x="Horizon (days)",
                y="MAE",
                color="Model",
                barmode="group",
                title="Mean Absolute Error by Forecast Horizon",
                hover_data=["Rows"]
            )
            mae_fig.update_layout(legend_title_text="Model")
            st.plotly_chart(mae_fig, use_container_width=True)

            rmse_fig = px.bar(
                evaluation_df,
                x="Horizon (days)",
                y="RMSE",
                color="Model",
                barmode="group",
                title="Root Mean Squared Error by Forecast Horizon",
                hover_data=["Rows"]
            )
            rmse_fig.update_layout(legend_title_text="Model")
            st.plotly_chart(rmse_fig, use_container_width=True)

            st.caption(
                "Lower MAE and RMSE are better. MAE measures typical absolute error; "
                "RMSE penalizes larger errors more strongly. Compare models at the same horizon."
            )
    else:
        st.info(
            "No backtest results were returned by the API. The forecast can still be viewed, "
            "but evaluation charts require the API response to include a 'backtest' section."
        )

    if not forecast_df.empty:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            product_options = sorted(forecast_df["product_id"].astype(str).unique())
            selected_products = st.multiselect(
                "Filter products",
                options=product_options,
                default=product_options[:min(5, len(product_options))],
                key="forecast_products"
            )
        with filter_col2:
            warehouse_options = sorted(forecast_df["warehouse"].astype(str).unique())
            selected_warehouses = st.multiselect(
                "Filter warehouses",
                options=warehouse_options,
                default=warehouse_options,
                key="forecast_warehouses"
            )

        filtered_forecast = forecast_df[
            forecast_df["product_id"].astype(str).isin(selected_products)
            & forecast_df["warehouse"].astype(str).isin(selected_warehouses)
        ].copy()

        if not filtered_forecast.empty:
            filtered_forecast["date"] = pd.to_datetime(filtered_forecast["date"])
            fig = px.line(
                filtered_forecast,
                x="date",
                y="predicted_demand",
                color="product_id",
                line_dash="warehouse",
                markers=True,
                title="Predicted Demand by Product and Warehouse",
                labels={
                    "date": "Forecast Date",
                    "predicted_demand": "Predicted Demand",
                    "product_id": "Product",
                    "warehouse": "Warehouse"
                }
            )
            fig.update_layout(legend_title_text="Product / Warehouse")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                filtered_forecast.sort_values(["date", "product_id", "warehouse"]),
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "Download Demand Forecast CSV",
                data=filtered_forecast.to_csv(index=False),
                file_name="ascip_demand_forecast.csv",
                mime="text/csv",
                key="download_demand_forecast"
            )
        else:
            st.info("No forecasts match the selected product and warehouse filters.")
    else:
        st.warning("The API returned no forecast records.")
else:
    st.info("Choose a forecast horizon and select Generate Demand Forecast to view predictions.")

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

st.divider()

st.subheader("🤖 ASCIP AI Supply Chain Assistant")
st.caption(
    "Ask questions about supplier performance, delivery delays, "
    "product demand, inventory, and stockout risks."
)

if "ascip_chat_history" not in st.session_state:
    st.session_state.ascip_chat_history = []

suggestions = [
    "Which suppliers have delivery or quality problems?",
    "Which products have the highest stockout risk?",
    "Explain the major supply chain risks.",
    "What should we prioritize to improve operations?"
]

st.markdown("**Try asking:**")
suggestion_cols = st.columns(2)

for i, suggestion in enumerate(suggestions):
    with suggestion_cols[i % 2]:
        if st.button(
            suggestion,
            key=f"chat_suggestion_{i}",
            use_container_width=True
        ):
            st.session_state.ascip_pending_question = suggestion

for message in st.session_state.ascip_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved data"):
                st.dataframe(
                    pd.DataFrame(message["sources"]),
                    use_container_width=True,
                    hide_index=True
                )

pending_question = st.session_state.pop(
    "ascip_pending_question", None
)

question = st.chat_input("Ask ASCIP a supply chain question...")

if pending_question or question:
    current_question = pending_question or question

    st.session_state.ascip_chat_history.append({
        "role": "user",
        "content": current_question
    })

    with st.chat_message("user"):
        st.markdown(current_question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing supply chain data..."):
            try:
                previous_messages = st.session_state.ascip_chat_history[:-1][-8:]
                history = [
                    {"role": message["role"], "content": message["content"]}
                    for message in previous_messages
                ]
                result = ask_ascip(current_question, history)
                answer = result.get(
                    "answer",
                    "The assistant returned no answer."
                )
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("View retrieved data"):
                        st.dataframe(
                            pd.DataFrame(sources),
                            use_container_width=True,
                            hide_index=True
                        )

                st.session_state.ascip_chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except requests.HTTPError as e:
                detail = e.response.text if e.response is not None else str(e)
                st.error(f"Chat API error: {detail}")

            except requests.RequestException as e:
                st.error(f"Could not connect to the chat API: {e}")

if st.session_state.ascip_chat_history:
    if st.button("🗑️ Clear chat history"):
        st.session_state.ascip_chat_history = []
        st.rerun()

st.caption(
    "ASCIP MVP | Synthetic supply-chain data | "
    "AI-assisted decision support, not live operational data"
)