# Autonomous Supply Chain Intelligence Platform (ASCIP)

ASCIP is an AI-driven, end-to-end supply chain intelligence platform designed to provide autonomous forecasting, risk intelligence, anomaly detection, inventory optimization, explainable insights (SHAP), and scenario simulation.

---

## 🏗️ Project Architecture
```text
ASCIP/
├── api/          # FastAPI backend services and endpoints
├── dashboard/    # User interface & visualization dashboard
├── data/         # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── documents/    # Unstructured supply chain documents for RAG
├── models/       # Trained ML/AI model binaries and artifacts
├── notebooks/    # Jupyter exploration and experimentation
├── rag/          # Vector store & RAG LLM decision assistant
├── src/          # Core reusable modular codebase
├── tests/        # Unit and integration test suites
├── .env          # Local environment variables
├── .gitignore    # Git ignored files and directories
├── requirements.txt # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.12.4
* Virtual environment (`.venv`)

### Installation & Execution
```bash
# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Verify dependencies
pip install -r requirements.txt
```
