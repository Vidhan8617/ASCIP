import json
import os
import pickle
import hashlib
import logging
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import sys
import faiss
from dotenv import load_dotenv
from google import genai
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ascip.rag")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None
embedding_model = None
vector_index = None
indexed_records = []


def load_supply_chain_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def get_dataset_hash():
    digest = hashlib.sha256()
    with open(DATA_PATH, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def get_cached_data():
    return load_supply_chain_data()


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


def build_vector_index():
    global embedding_model, vector_index, indexed_records

    started = time.perf_counter()
    df = get_cached_data().fillna("")
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
                logger.info("RAG index loaded in %.2fs", time.perf_counter() - started)
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
        pickle.dump({"dataset_hash": current_hash, "records": indexed_records}, file)

    logger.info("RAG index built in %.2fs", time.perf_counter() - started)


def ensure_index():
    if vector_index is None or embedding_model is None:
        build_vector_index()


def retrieve_relevant_records(question, limit=5):
    ensure_index()
    started = time.perf_counter()
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

    logger.info("Semantic retrieval completed in %.2fs", time.perf_counter() - started)
    return results


@lru_cache(maxsize=1)
def get_supplier_summary_cached():
    result = calculate_supplier_risk(get_cached_data().copy())
    if "risk_score" in result.columns:
        result = result.sort_values("risk_score", ascending=False)
    return result.to_dict(orient="records")


@lru_cache(maxsize=1)
def get_inventory_summary_cached():
    result = analyze_inventory(get_cached_data().copy())
    if "suggested_reorder_qty" in result.columns:
        result = result.sort_values("suggested_reorder_qty", ascending=False)
    return result.to_dict(orient="records")


def get_supplier_summary():
    return get_supplier_summary_cached()


def get_inventory_summary():
    return get_inventory_summary_cached()


def select_context(question, history=None):
    history = history or []
    history_text = " ".join(
        str(message.get("content", ""))
        for message in history[-6:]
        if isinstance(message, dict)
    )
    combined_question = f"{history_text} {question}".lower()
    retrieval_query = f"{history_text} {question}".strip() if history_text else question
    context = {"semantic_records": retrieve_relevant_records(retrieval_query)}

    supplier_terms = ["supplier", "vendor", "delivery", "late", "defect", "quality", "on-time", "on time", "supplier risk", "risk score"]
    inventory_terms = ["inventory", "stockout", "stock out", "reorder", "shortage", "warehouse", "demand", "stock", "product"]

    if any(term in combined_question for term in supplier_terms):
        context["supplier_risk_summary"] = get_supplier_summary()[:10]

    if any(term in combined_question for term in inventory_terms):
        context["inventory_summary"] = get_inventory_summary()[:10]

    return context


def answer_question(question, history=None):
    question = question.strip()

    if not question:
        return {"answer": "Please enter a question.", "sources": [], "status": "empty_question"}

    if client is None:
        return {"answer": "Gemini API key is missing. Check GEMINI_API_KEY in your .env file.", "sources": [], "status": "missing_api_key"}

    started = time.perf_counter()

    try:
        stage = time.perf_counter()
        context = select_context(question, history)
        logger.info("Context selection completed in %.2fs", time.perf_counter() - stage)

        recent_history = [
            {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))[:1000]}
            for message in (history or [])[-6:]
            if isinstance(message, dict) and message.get("content")
        ]

        prompt = f"""
You are ASCIP, an AI-powered supply-chain intelligence assistant.
Answer the current question using only the provided dataset evidence and conversation context.
Use supplier_risk_summary for supplier-wide comparisons and risk levels.
Use inventory_summary for stockout risks and reorder recommendations.
Use semantic_records for specific daily observations and supporting examples.
Resolve references such as "it", "that supplier", and "that risk" using conversation history when possible.
Do not treat an individual row as proof of overall supplier performance.
Do not invent figures, suppliers, products, or conclusions.
If evidence is insufficient, clearly state the limitation.
Give concise, practical recommendations supported by the data.

Recent conversation:
{json.dumps(recent_history, ensure_ascii=False)}

Current user question:
{question}

Retrieved evidence:
{json.dumps(context, default=str)}
"""
        stage = time.perf_counter()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        logger.info("Gemini generation completed in %.2fs", time.perf_counter() - stage)

        answer = (response.text or "").strip() or "Gemini returned an empty response. Please try again."
        sources = []
        for key, value in context.items():
            if isinstance(value, list):
                sources.extend({"source_type": key, **record} for record in value)

        logger.info("Total chatbot request completed in %.2fs", time.perf_counter() - started)
        return {"answer": answer, "sources": sources, "status": "success"}

    except Exception as e:
        logger.exception("Chatbot request failed")
        return {"answer": f"Unable to answer the question right now: {str(e)}", "sources": [], "status": "error"}
