 # News Headline Classification

 An end-to-end mini project that classifies news headlines into categories using both a traditional ML pipeline (TF-IDF + Logistic Regression) and a zero-shot transformer model, exposed via Streamlit UI and optional Flask backend API.

 Live demo: https://news-headline-classification.streamlit.app/

 Created by Team **Devansh Tiwari (Team Leader), Ayush Singh, Aridaman Singh, Atul Kr. Singh** under the supervision of **Dr. Imtiyaz Ahmed** and coordination of **Mr. Abhishek Suraj Sir**.

 ---

 ## Tech Stack
- Python (3.x)
- Streamlit (frontend UI)
- Flask (backend API)
- scikit-learn (classical ML)
- transformers + torch (zero-shot transformer model)
- pandas, numpy (data handling)
- joblib (model persistence)

 ### Key Package Versions (see `requirements.txt`)
- pandas
- numpy
- scikit-learn
- flask
- streamlit
- requests
- joblib
- transformers
- torch

 ---

 ## Project Structure
- `streamlit_app.py`: Streamlit UI; can run model locally or call backend API.
- `app.py`: Flask API (`/predict`, `/health`) serving zero-shot classifier + heuristics.
- `data_prep.py`: Prepares a small CSV (`news_headlines.csv`) from the Kaggle News Category dataset.
- `train_model.py`: Trains TF-IDF + Logistic Regression model; saves to `models/`.
- `models/`: Saved sklearn pipeline and labels.
- `requirements.txt`: Dependencies list.

 ---

 ## Models Used
1) **Zero-shot Transformer (inference-time, no finetuning)**
   - Primary models: `valhalla/distilbart-mnli-12-3` (preferred, lightweight), fallback `facebook/bart-large-mnli`.
   - Why: Handles arbitrary label sets without retraining; good for small projects / rapid iteration.
   - Usage:
     - Streamlit local mode (`use_local`): runs the pipeline directly.
     - Flask backend (`/predict`): serves the same pipeline.
   - Heuristics: keyword-based override when model confidence is low (<0.35) to improve obvious cases.

2) **Classical ML Baseline**
   - Pipeline: `TfidfVectorizer(max_features=5000, ngram_range=(1,2))` + `LogisticRegression(max_iter=1000)`.
   - Training script: `train_model.py`.
   - Saved artifacts: `models/tfidf_logreg_pipeline.joblib`, `models/labels.joblib`.
   - Why: Fast, lightweight baseline; transparent and easy to debug; good offline fallback.

 ---

 ## Data Preparation (`data_prep.py`)
 - Input: `News_Category_Dataset_v3.json` (or `News_Category_Dataset.json`) from Kaggle.
 - Steps:
   1. Load JSONL, extract `headline` and `category`.
   2. Drop NA/duplicates.
   3. Sample ~250 rows for a lightweight demo dataset.
   4. Save to `news_headlines.csv`.
 - Reasoning: Keeps dataset small for quick experimentation while preserving category diversity.

 ---

 ## Training (`train_model.py`)
 - Loads `news_headlines.csv`, normalizes labels into: `{tech, health, business, entertainment, sports, politics}`.
 - Cleans text: lowercase, strip non-letters, collapse spaces.
 - Splits train/test (approx. 200/50, stratified when safe).
 - Fits TF-IDF + Logistic Regression.
 - Evaluates (accuracy, classification report).
 - Saves pipeline and label set to `models/`.
 - Reasoning: Provides a reproducible classical model and artifacts usable as a fallback or comparison to the zero-shot approach.

 ---

 ## Inference Flow
 Textual flow (UI local mode):
 ```
 User -> Streamlit text input
      -> clean_text + keyword heuristics
      -> zero-shot classifier (distilbart-mnli) on LABELS
      -> confidence check; heuristic override if low
      -> display predicted_category (+ confidence)
 ```
 Textual flow (UI calling backend):
 ```
 User -> Streamlit -> POST /predict (Flask)
                    -> Flask: clean_text + heuristics + zero-shot
                    -> response: predicted_category, confidence
                    -> Streamlit renders result
 ```

 ---

 ## Running the Project
 1) Install dependencies (recommend venv):
    ```
    pip install -r requirements.txt
    ```
 2) Run Streamlit UI (local model by default):
    ```
    streamlit run streamlit_app.py
    ```
 3) (Optional) Run Flask backend instead of local model:
    ```
    python app.py
    ```
    Set environment variable in Streamlit before running:
    ```
    set BACKEND_URL=http://127.0.0.1:8000/predict   # Windows PowerShell: $env:BACKEND_URL="http://127.0.0.1:8000/predict"
    ```
    Then open the Streamlit app and uncheck “Run model inside this app”.

 ---

 ## Deployment Notes
 - Streamlit Cloud-friendly: local zero-shot mode avoids needing a separate backend service.
 - Backend option: suitable for containerization or API integration.
 - Model download: first zero-shot call may take ~30–60s to download weights.

 ---

 ## Credits
 - Team: Devansh Tiwari (Team Leader), Ayush Singh, Aridaman Singh, Atul Kr. Singh
 - Supervision: Dr. Imtiyaz Ahmed
 - Coordination: Mr. Abhishek Suraj
 - Live app: https://news-headline-classification.streamlit.app/

