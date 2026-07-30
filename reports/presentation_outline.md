# Smart Multilingual Sentiment Analysis - Presentation Slides Outline

## Slide 1: Title Slide
* **Title:** Smart Multilingual Product Review Sentiment Analysis
* **Subtitle:** Fine-Tuned Cross-lingual Transformer with a React (Vite) Frontend
* **Presenter:** Senior AI & NLP Engineer
* **Visuals:** Modern neural network icon, project logo, tech badges (PyTorch, Transformers, React, FastAPI)

---

## Slide 2: Problem Statement
* **Key Challenges:**
  * E-commerce reviews are increasingly multilingual and code-mixed.
  * Roman Urdu (Urdu written in Latin characters) is highly colloquial, has inconsistent spelling, and lacks standard tokenizers.
  * Native scripts (Arabic for Urdu, Devanagari for Hindi) require specialized Normalizers.
  * Multiple models are expensive to host; a unified solution is required.

---

## Slide 3: Project Goals
* **Core Objectives:**
  * Combine Amazon, Flipkart, Urdu, and Hindi sentiment datasets into one standardized format.
  * Develop script-aware language routing.
  * Fine-tune **XLM-RoBERTa (base)** on a single dataset to handle all 4 languages.
  * Build a persistent database using SQLite.
  * Create a simple, clean, and professional React Single Page Application (SPA).

---

## Slide 4: Technology Stack
* **Deep Learning:** PyTorch, Hugging Face Transformers, Accelerate, Datasets, Scikit-learn
* **NLP & Processing:** NLTK, langdetect, Custom Normalization tables
* **Frontend UI:** React 18 (Vite), Recharts, Lucide React, clean CSS
* **Backend API & DB:** FastAPI, Uvicorn, SQLite, ReportLab (PDF generation)

---

## Slide 5: System Architecture
* **Data Flow Diagram:**
  * React SPA input $\rightarrow$ HTTP POST to FastAPI $\rightarrow$ Language Detection Heuristic $\rightarrow$ Preprocessing Normalization $\rightarrow$ tokenization $\rightarrow$ PyTorch forward pass $\rightarrow$ Softmax probability $\rightarrow$ SQLite log $\rightarrow$ JSON Response $\rightarrow$ Recharts graphs.
* **Fast Startup Heuristic:** Uses local caching for the model or fallback lexicon engine to guarantee uvicorn boot times of under 1 second.

---

## Slide 6: Preprocessing & normalizations
* **Cleaning steps:**
  * Removes HTML tags and URLs.
  * Unicode normalization (NFKC).
  * Collapses multi-spaces.
  * Converts English/Roman Urdu to lowercase.
* **Script-Aware Processing:**
  * Resolves Urdu character mappings (e.g. Arabic Kaf and Ya mapped to Urdu).
  * Hindi Devanagari character verification.

---

## Slide 7: Model & Training Pipeline
* **Base Model:** XLM-RoBERTa (base) - pre-trained on 2.5TB of text across 100 languages.
* **Hyperparameters:**
  * Epochs: 3
  * Batch Size: 8
  * Learning Rate: $2.0 \times 10^{-5}$
  * Target Labels: Negative (0), Neutral (1), Positive (2)
* **Local CPU Mode:** Features a `--quick` train parameter to verify setup on a sub-dataset in under 2 minutes.

---

## Slide 8: Interactive Web App Features
* **Multi-Tab Dashboard Pages:**
  * **Home:** Platform overview and quick start.
  * **Analyze Review:** Real-time prediction with word color-coding.
  * **Bulk Upload:** CSV drag-and-drop batch classification.
  * **Dashboard:** Recharts line, bar, donut diagrams and keyword clouds.
  * **History:** Search and PDF report exports.
  * **Model Details:** Validation loss/accuracy tables.

---

## Slide 9: Evaluation Results
* **Performance Metrics:**
  * Baseline accuracy: $85.20\%$
  * F1-score: $84.9\%$
  * Model outputs: Confidences and Latency in milliseconds.
* **Outputs Saved:** Loss history plot, accuracy curve, confusion matrix heatmap, classification report JSON/TXT.

---

## Slide 10: Future Roadmap
* **Aspect-Based Analysis:** Output sentiment per feature (e.g., screen vs battery).
* **Mixed-Script Support:** Specifically target code-mixed sentences (e.g., Hinglish).
* **Feedback Loop:** User-guided labeling interface to re-train checkpoints.

---

## Slide 11: Q&A / Thank You
* **Summary:** Successfully built a local, open-source, React + FastAPI multilingual sentiment analysis platform.
* **Contact & Links:** README.md, Project Report.
