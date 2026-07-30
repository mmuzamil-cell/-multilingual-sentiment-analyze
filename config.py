import os
import logging
from pathlib import Path

# Base workspace directory
BASE_DIR = Path(__file__).resolve().parent

# Directory definitions
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
DB_DIR = BASE_DIR / "database"
REPORT_DIR = BASE_DIR / "reports"
EXPORT_DIR = BASE_DIR / "exports"
EVAL_DIR = BASE_DIR / "evaluation"

# Create directories if they do not exist
for folder in [DATA_DIR, MODEL_DIR, LOG_DIR, DB_DIR, REPORT_DIR, EXPORT_DIR, EVAL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Database Config
DB_PATH = DB_DIR / "sentiment_analysis.db"

# Model Config
# Fine-tune XLM-RoBERTa (base model)
MODEL_NAME = "xlm-roberta-base"
FINE_TUNED_MODEL_DIR = MODEL_DIR / "xlm_roberta_sentiment"

# Training Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

# Supported Languages Mapping
LANGUAGES = {
    "en": "English",
    "ur": "Urdu",
    "ur_roman": "Roman Urdu",
    "unknown": "Unknown"
}

# Sentiments Mapping
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
LABEL_TO_ID = {v: k for k, v in SENTIMENT_LABELS.items()}

# Set up logging configuration
LOG_FILE = LOG_DIR / "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance with the specified name."""
    return logging.getLogger(name)
