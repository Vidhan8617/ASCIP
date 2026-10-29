# ASCIP — AI-Powered Supply Chain Intelligence Platform

An AI-powered supply chain analytics platform combining demand forecasting, supplier risk assessment, inventory intelligence, and a retrieval-augmented generation (RAG) assistant to support data-driven operational decisions.

## Overview

ASCIP helps analyze supply chain operations by combining machine learning, data analytics, and generative AI in one application.

The platform provides an interactive Streamlit dashboard backed by a FastAPI service. It supports demand forecasting, supplier risk analysis, inventory risk classification, reorder recommendations, and natural-language queries about supply chain data.

## Key Features

### 1. Demand Forecasting

- Random Forest regression for demand prediction.
- Forecasts across product–warehouse combinations.
- Configurable forecast horizons.
- Interactive forecast charts, filters, and CSV exports.
- MAE and RMSE evaluation.
- Comparison against previous-day and seven-day seasonal baselines.

### 2. Inventory Intelligence

- Inventory risk classification: High, Medium, and Low.
- Demand and lead-time-based reorder point calculation.
- Safety stock estimation.
- Suggested reorder quantities.
- Inventory reports and CSV downloads.

### 3. Supplier Risk Analysis

- Supplier risk analysis using supply chain operational data.
- Supplier risk summaries and interactive dashboard visualizations.

### 4. Generative AI Insights

- Gemini-powered supplier and inventory insights.
- Natural-language explanations to support operational analysis.

### 5. RAG-Based Supply Chain Assistant

- Semantic retrieval using Sentence Transformers and FAISS.
- Retrieval from supply chain records.
- Contextual responses using Gemini when configured.
- Natural-language questions about supplier performance, inventory, demand, and stockout risks.

### 6. API and Dashboard

- FastAPI backend exposing supply chain analytics endpoints.
- Streamlit dashboard for interactive exploration.
- Plotly visualizations.
- REST API communication and downloadable reports.

## Technology Stack

| Category          | Technologies                        |
| ----------------- | ----------------------------------- |
| Language          | Python                              |
| Data processing   | Pandas, NumPy                       |
| Machine learning  | Scikit-learn, Random Forest         |
| Backend           | FastAPI, Uvicorn                    |
| Dashboard         | Streamlit                           |
| Visualization     | Plotly                              |
| Generative AI     | Google Gemini API, Google GenAI SDK |
| Retrieval         | Sentence Transformers, FAISS        |
| API communication | REST API, Requests                  |
| Version control   | Git, GitHub                         |

## Architecture

```text
Supply Chain Dataset (CSV)
          |
          v
Data Processing and Intelligence Modules
          |
          +-------------------+
          |                   |
          v                   v
Demand Forecasting      Supplier and Inventory
Random Forest           Risk Intelligence
          |                   |
          +---------+---------+
                    |
                    v
              FastAPI Backend
                    |
                    v
             Streamlit Dashboard
                    |
          +---------+----------+
          |                    |
          v                    v
    Visual Analytics      AI and RAG Assistant
                               |
                       Sentence Transformers
                               |
                         FAISS Retrieval
                               |
                         Gemini API*
```

*Gemini-powered responses depend on valid API configuration.*

## Forecast Evaluation

ASCIP evaluates Random Forest predictions using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

The backtesting module compares Random Forest against:

- Previous-day demand baseline.
- Seven-day seasonal baseline.

The implemented backtest evaluates 7-, 14-, and 30-day horizons. The current configuration uses one backtest origin, so these results should be treated as preliminary evidence rather than proof of general performance.

The chronological holdout evaluation uses observed lag features, while the multi-day forecast recursively uses its own predictions. These evaluation methods are not directly interchangeable.

## Inventory Methodology

The inventory intelligence module estimates:

- Average daily demand over a recent 14-day window.
- Estimated lead time from historical delivery data.
- Safety stock based on demand variability and lead time.
- Reorder point from expected lead-time demand plus safety stock.
- Suggested reorder quantity to reach a target covering lead time and an additional seven days.

These calculations are heuristic planning recommendations. Their accuracy depends on the quality and representativeness of the available data and assumptions.

## Project Structure

```text
ASCIP/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── src/
│   ├── supplier_risk.py
│   ├── inventory_intelligence.py
│   ├── ai_insights.py
│   ├── rag_chatbot.py
│   └── demand_forecasting.py
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
├── .env
└── README.md
```

## Setup and Execution

### Prerequisites

- Python 3.12
- Git
- A configured Gemini API key for Gemini-powered functionality

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ASCIP
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

Never commit API keys or other secrets to GitHub.

### 5. Start the API

```powershell
python -m uvicorn api.main:app --reload
```

API base URL: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

### 6. Start the dashboard

Open a second terminal, activate the same virtual environment, and run:

```powershell
python -m streamlit run dashboard/app.py
```

Dashboard URL: `http://localhost:8501`

Run the API and dashboard simultaneously.

## Known Limitations

- Forecast backtesting currently uses one origin.
- Forecast accuracy may vary by product, warehouse, and forecast horizon.
- The forecasting evaluation with observed lag features differs from recursive multi-day forecasting.
- Inventory recommendations use simplified demand and lead-time assumptions.
- Results depend on dataset quality and representativeness.
- Gemini-powered features require appropriate API configuration and network access.
- The current project is a decision-support prototype, not a production-validated supply chain optimization system.

## Future Enhancements

- Redis caching.
- Kafka-based event streaming.
- Databricks and PySpark for scalable data processing.
- Power BI reporting.
- Automated testing and continuous integration.
- Production deployment and monitoring.
- Expanded multi-origin forecasting evaluation.

## Security

- Store API credentials in environment variables.
- Exclude `.env`, `.venv/`, and Python cache files from version control.
- Never publish private credentials or sensitive operational data.

## Project Status

ASCIP currently combines machine-learning forecasting, supplier and inventory analytics, a FastAPI backend, a Streamlit dashboard, and a Gemini-powered RAG assistant. Further validation, documentation, and production hardening remain future work.

## License

Add the appropriate license before distributing the project publicly.

