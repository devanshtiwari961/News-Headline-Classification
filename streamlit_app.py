import streamlit as st
import requests
import os
from transformers import pipeline

st.set_page_config(page_title="News Headline Classification", page_icon="📰", layout="centered")

st.title("📰 News Headline Classification")
st.write("Enter a news headline to predict its category.")

# Target labels for classification
LABELS = ["tech", "health", "business", "entertainment", "sports", "politics"]

# Cleaning util (same as backend)
import re

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Simple keyword heuristics to improve classification for obvious cases
HEURISTIC_KEYWORDS = {
    "tech": [
        "tech", "technology", "software", "hardware", "iphone", "android", "apple", "google",
        "microsoft", "ai", "robot", "device", "gadget", "update", "launch"
    ],
    "health": [
        "health", "hospital", "medical", "doctor", "medicine", "emergency", "covid", "vaccine",
        "patient", "clinic"
    ],
    "business": [
        "business", "finance", "company", "earnings", "profit", "stock", "market", "invest",
        "shares", "merger", "acquisition", "revenue"
    ],
    "entertainment": [
        "movie", "film", "series", "tv", "celebrity", "actor", "actress", "bollywood", "hollywood",
        "music", "album", "song", "trailer"
    ],
    "sports": [
        "sport", "sports", "team", "match", "game", "championship", "league", "score", "win",
        "goal", "tournament", "cricket", "football", "soccer", "basketball"
    ],
    "politics": [
        "politics", "government", "prime minister", "president", "election", "party", "policy",
        "parliament", "bill", "vote"
    ],
}

def rule_based_category(text: str):
    counts = {}
    for cat, kws in HEURISTIC_KEYWORDS.items():
        for kw in kws:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                counts[cat] = counts.get(cat, 0) + 1
    if counts:
        return max(counts, key=lambda k: counts[k])
    return None

# Lazy load zero-shot classifier
_zs = None

def get_zs_classifier():
    global _zs
    if _zs is None:
        try:
            _zs = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
        except Exception:
            _zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _zs

# Avoid Streamlit secrets when not configured; fall back to env var or default
backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000/predict")

# st.caption(f"Backend URL: {backend_url}")

use_local = st.checkbox("Run model inside this app (recommended for Streamlit Cloud)", value=True)

headline = st.text_input("News Headline", "")

if st.button("Predict"):
    if not headline.strip():
        st.warning("Please enter a headline.")
    else:
        try:
            with st.spinner("Running model... first call may download a pretrained model (~30-60s)."):
                if use_local:
                    cleaned = clean_text(headline)
                    heur = rule_based_category(cleaned)
                    zs = get_zs_classifier()(cleaned, LABELS, multi_label=False)
                    pred = zs["labels"][0]
                    conf = float(zs["scores"][0])
                    if heur and conf < 0.35:
                        pred = heur
                    data = {"predicted_category": pred, "confidence": conf}
                else:
                    resp = requests.post(backend_url, json={"headline": headline}, timeout=60)
                    if resp.status_code != 200:
                        st.error(f"Backend error ({resp.status_code}): {resp.text}")
                        st.stop()
                    data = resp.json()
            st.success(f"Predicted Category: {data.get('predicted_category')}")
            conf = data.get("confidence")
            if conf is not None:
                st.write(f"Confidence: {conf:.2f}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. If deploying on Streamlit Cloud, enable 'Run model inside this app'.")
        except Exception as e:
            st.error(f"Request failed: {e}")

if use_local:
    st.caption("Model: Zero-shot (Hugging Face Transformers) + keyword heuristics")
else:
    st.caption("Model: Served by backend API at BACKEND_URL")
