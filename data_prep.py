import pandas as pd
import json
import os

"""
Prepare a small CSV dataset (news_headlines.csv) from Kaggle News Category Dataset.
- Expected input file: News_Category_Dataset_v3.json (JSON lines), with fields: 'category' and 'headline'
- Output file: news_headlines.csv with columns: headline, category (sample ~250 rows)
Place the Kaggle JSON file in the project root before running.
"""

INPUT_JSON_CANDIDATES = [
    "News_Category_Dataset_v3.json",
    "News_Category_Dataset.json",
]
OUTPUT_CSV = "news_headlines.csv"
SAMPLE_SIZE = 250


def find_input_json():
    for fname in INPUT_JSON_CANDIDATES:
        if os.path.exists(fname):
            return fname
    raise FileNotFoundError(
        "Kaggle dataset JSON not found. Please place 'News_Category_Dataset_v3.json' in the project root."
    )


def main():
    in_path = find_input_json()
    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                headline = obj.get("headline", "").strip()
                category = obj.get("category", "").strip()
                if headline and category:
                    rows.append({"headline": headline, "category": category})
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(rows)
    # Drop duplicates and NA
    df = df.dropna(subset=["headline", "category"]).drop_duplicates(subset=["headline"]).copy()

    # Sample for mini project
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()