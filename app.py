from flask import Flask, request, jsonify
import os
import re
from transformers import pipeline

LABELS = ["tech", "health", "business", "entertainment", "sports", "politics"]

app = Flask(__name__)

# Lazy load zero-shot classifier
zs_classifier = None

# Add same cleaning used during training
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
            # word boundary search; kw may contain space (e.g., prime minister)
            if re.search(r"\\b" + re.escape(kw) + r"\\b", text):
                counts[cat] = counts.get(cat, 0) + 1
    if counts:
        # pick the category with the most keyword hits
        return max(counts, key=lambda k: counts[k])
    return None


def get_zs_classifier():
    global zs_classifier
    if zs_classifier is None:
        # Use a small, widely available model suitable for zero-shot classification
        # "valhalla/distilbart-mnli-12-3" or "facebook/bart-large-mnli"; choose distilbart for lighter footprint
        try:
            zs_classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
        except Exception:
            # Fallback to bart-large-mnli if the above is unavailable
            zs_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return zs_classifier

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    headline = data.get("headline", "")
    if not isinstance(headline, str) or not headline.strip():
        return jsonify({"error": "Invalid input. Provide a non-empty 'headline'"}), 400

    cleaned = clean_text(headline)

    # Heuristic prediction for obvious cases
    heuristic_cat = rule_based_category(cleaned)

    # Zero-shot classification
    classifier = get_zs_classifier()
    zs = classifier(cleaned, LABELS, multi_label=False)
    pred = zs["labels"][0]
    proba = float(zs["scores"][0])

    # If model is uncertain, prefer heuristic classification when available
    if heuristic_cat and proba < 0.35:
        pred = heuristic_cat

    return jsonify({"predicted_category": pred, "confidence": proba})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)