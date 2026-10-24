import json
import os
import pickle
import hashlib
import pandas as pd
import sys
import faiss
from dotenv import load_dotenv
from google import genai
from pathlib import Path
from sentence_transformers import SentenceTransformer

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")) if str(ROOT / "src") not in sys.path else None

from supplier_risk import calculate_supplier_risk
from inventory_intelligence import analyze_inventory

DATA_PATH = ROOT / "data" / "raw" / "supply_chain_data.csv"
INDEX_DIR = ROOT / "data" / "processed" / "rag_index"
INDEX_PATH = INDEX_DIR / "supply_chain.faiss"
RECORDS_PATH = INDEX_DIR / "records.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.5-flash-lite"

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None
embedding_model = None
vector_index = None
indexed_records = []


def load_supply_chain_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def record_to_text(row):
    return (
        f"Date {row.get('date', '')}; "
        f"product {row.get('product_name', '')}; "
        f"product ID {row.get('product_id', '')}; "
        f"category {row.get('category', '')}; "
        f"warehouse {row.get('warehouse', '')}; "
        f"supplier {row.get('supplier_id', '')}; "
        f"demand {row.get('demand', '')}; "
        f"inventory {row.get('inventory', '')}; "
        f"order quantity {row.get('order_quantity', '')}; "
        f"delivery days {row.get('delivery_days', '')}; "
        f"defect rate {row.get('defect_rate', '')}; "
        f"on-time indicator {row.get('supplier_on_time', '')}; "
        f"stockout {row.get('stockout', '')}"
    )


def get_dataset_hash():
    digest = hashlib.sha256()
    with open(DATA_PATH, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_vector_index():
    global embedding_model, vector_index, indexed_records

    df = load_supply_chain_data().fillna("")
    current_hash = get_dataset_hash()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists() and RECORDS_PATH.exists():
        try:
            with open(RECORDS_PATH, "rb") as file:
                saved = pickle.load(file)

            if saved.get("dataset_hash") == current_hash:
                vector_index = faiss.read_index(str(INDEX_PATH))
                indexed_records = saved["records"]
                embedding_model = SentenceTransformer(MODEL_NAME)
                return
        except (OSError, ValueError, pickle.PickleError, EOFError, KeyError):
            pass

    embedding_model = SentenceTransformer(MODEL_NAME)
    indexed_records = df.to_dict(orient="records")
    texts = [record_to_text(row) for row in indexed_records]

    embeddings = embedding_model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    vector_index = faiss.IndexFlatIP(embeddings.shape[1])
    vector_index.add(embeddings)
    faiss.write_index(vector_index, str(INDEX_PATH))

    with open(RECORDS_PATH, "wb") as file:
        pickle.dump({
            "dataset_hash": current_hash,
            "records": indexed_records
        }, file)


def ensure_index():
    if vector_index is None or embedding_model is None:
        build_vector_index()


def retrieve_relevant_records(question, limit=8):
    ensure_index()

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    count = min(limit, len(indexed_records))
    scores, indices = vector_index.search(query_embedding, count)
    results = []

    for score, index in zip(scores[0], indices[0]):
        if index >= 0:
            record = indexed_records[int(index)].copy()
            record["_similarity"] = round(float(score), 4)
            results.append(record)

    return results


def get_supplier_summary():
    df = load_supply_chain_data()
    return calculate_supplier_risk(df).to_dict(orient="records")


def get_inventory_summary():
    df = load_supply_chain_data()
    return analyze_inventory(df).to_dict(orient="records")


def select_context(question):
    q = question.lower()
    context = {"semantic_records": retrieve_relevant_records(question)}

    supplier_terms = [
        "supplier", "vendor", "delivery", "late", "defect",
        "quality", "on-time", "on time", "supplier risk"
    ]
    inventory_terms = [
        "inventory", "stockout", "stock out", "reorder",
        "shortage", "warehouse", "demand", "stock"
    ]

    if any(term in q for term in supplier_terms):
        context["supplier_risk_summary"] = get_supplier_summary()

    if any(term in q for term in inventory_terms):
        context["inventory_summary"] = get_inventory_summary()

    return context


def answer_question(question):
    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "status": "empty_question"
        }

    if client is None:
        return {
            "answer": "Gemini API key is missing. Check GEMINI_API_KEY in your .env file.",
            "sources": [],
            "status": "missing_api_key"
        }

    try:
        context = select_context(question)

        prompt = f"""
You are ASCIP, an AI-powered supply-chain intelligence assistant.

Answer the user's question using only the provided dataset evidence.
Use supplier_risk_summary for supplier-wide comparisons and risk levels.
Use inventory_summary for stockout risks and reorder recommendations.
Use semantic_records for specific daily observations and supporting examples.
Do not treat an individual row as proof of overall supplier performance.
Do not invent figures, suppliers, products, or conclusions.
If evidence is insufficient, clearly state the limitation.
Give practical recommendations supported by the data.

User question:
{question}

Retrieved evidence:
{json.dumps(context, default=str)}
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        answer = (response.text or "").strip()

        if not answer:
            answer = "Gemini returned an empty response. Please try again."

        sources = []
        for key, value in context.items():
            if isinstance(value, list):
                sources.extend(
                    {"source_type": key, **record}
                    for record in value
                )

        return {
            "answer": answer,
            "sources": sources,
            "status": "success"
        }

    except Exception as e:
        return {
            "answer": f"Unable to answer the question right now: {str(e)}",
            "sources": [],
            "status": "error"
        }