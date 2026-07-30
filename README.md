# Smart Multilingual Product Review Sentiment Analysis

A production-quality machine learning and natural language processing (NLP) application designed to classify product reviews into **Positive**, **Neutral**, or **Negative** sentiments. 

The platform supports and automatically detects **English**, **Urdu**, **Roman Urdu** (transliterated Urdu written in the Latin alphabet), and **Hindi** within a single integrated model.

---

## 🚀 Key Features

* **Multi-Language Detection & Cleaning:** Dynamic preprocessing pipeline tailored to English, Hindi (Devanagari script), Urdu (Arabic/Perso-Arabic script), and Roman Urdu.
* **Fine-Tuned Cross-lingual Transformer:** Utilizes a fine-tuned **XLM-RoBERTa (base)** sequence classifier built in PyTorch to achieve robust semantics representation.
* **Lexicon Fallback Engine:** Features a fallback heuristic lexicon logic to ensure instant classification capability even if deep learning checkpoints are compiling/loading.
* **React Web App Dashboard:** Simple, clean, and professional interface built in React (Vite) styled with Stripe-inspired card frameworks, forms, and responsive grid layouts.
* **FastAPI Backend Server:** Python API service exposing REST endpoints to execute text cleaning, classification, and database queries.
* **Batch CSV Analysis:** Bulk upload reviews to classify them in parallel, calculate aggregates, and download predictions as CSV.
* **Executive Report PDF Generator:** Auto-compiles predicted reviews, average confidence indices, and sentiment breakdowns into a professionally styled PDF layout (ReportLab).
* **Interactive Browser Charts:** High-fidelity Recharts timelines, pie charts, and language bar charts, coupled with keyword tag clouds.
* **Persistent History:** SQLite-backed prediction logging to search, query, and filter historical reviews.

---

## 🛠 Technology Stack

* **Backend API & Core ML:** Python 3.12+, FastAPI, Uvicorn, PyTorch, Hugging Face Transformers, Scikit-learn
* **NLP & Processing:** NLTK, langdetect, Custom Unicode script normalization tables
* **UI & Visualizations:** React 18, Vite, Recharts, Lucide React, clean CSS
* **Storage & Exporters:** SQLite, ReportLab (PDF)

---

## 📂 Project Structure

```text
├── backend/                    # FastAPI web server and API endpoints
├── data/                       # Standardized datasets and splits (Train, Val, Test)
├── models/                     # Checkpoints for fine-tuned XLM-RoBERTa
├── training/                   # Model training and dataset builder scripts
├── preprocessing/              # Cleaning scripts, normalizers, and language detector
├── evaluation/                 # Loss/Accuracy curves, confusion matrices, class reports
├── database/                   # SQLite database configurations and DBManager
├── utils/                      # Inference pipeline, PDF/CSV report exporters
├── frontend/                   # React Single Page Application (SPA)
│   ├── src/                    # App.jsx, main.jsx, and clean style.css
│   └── package.json            # Frontend node packages
├── reports/                    # Final project report and presentation slides outline
├── requirements.txt            # Python dependencies list
└── README.md                   # Setup and usage guide
```

---

## 💻 Installation & Setup

### 1. Clone the Workspace
Ensure you have cloned this repository onto your system and navigated into the workspace folder.

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Node.js Dependencies
Navigate into the `frontend` folder and run:
```bash
cd frontend
npm install
```

---

## 🏋️ Model Training

The system contains an automated dataset compiler that either maps existing CSV files or generates a standard multilingual review dataset.

### 1. Standardize and Build Datasets
Compile the standardized train, validation, and test splits into the `data/` folder:
```bash
python -m training.dataset_builder
```

### 2. Fine-tune XLM-RoBERTa
To start the model training on your local system, run:
```bash
python -m training.train
```

*Note: To quickly test that the Hugging Face and PyTorch pipelines run successfully without waiting for a full local epoch, run in verification mode:*
```bash
python -m training.train --quick
```
This loads a small subset of data, completes 1 verification epoch, and outputs model metrics and curves directly in `evaluation/`.

---

## 🖥️ Running the Web Application

To launch and run the entire application, you need to run the **Backend API** and the **Frontend dev server** concurrently.

### 1. Start the FastAPI Backend
Open a terminal in the root directory and run:
```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```
This starts the backend API on `http://127.0.0.1:8000`.

### 2. Start the React Frontend
Open a separate terminal in the root directory and run:
```bash
cd frontend
npm run dev
```
This boots the React Vite dev server on `http://localhost:5173`. Open this URL in your web browser to access the dashboard!

---

## 🔍 Verification

You can execute a quick end-to-end backend test covering database insertions, language detection, inference lookups, and report exports:
```bash
python scratch/verify_backend.py
```
This runs assertions across all core systems to verify correctness.
