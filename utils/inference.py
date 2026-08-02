import os
import re
import time
import numpy as np
from pathlib import Path

from config import (
    FINE_TUNED_MODEL_DIR, 
    MODEL_NAME, 
    MAX_LEN, 
    SENTIMENT_LABELS, 
    get_logger
)
from preprocessing.cleaner import preprocess_review, tokenize_text, get_cleaned_words
from preprocessing.lang_detector import detect_language

logger = get_logger("inference")

# Lexicons for highlight keywords and fallback prediction
LEXICONS = {
    "en": {
        "positive": {"great", "amazing", "excellent", "love", "best", "good", "satisfied", "perfect", "superb", "awesome", "nice", "wonderful", "happy", "fast", "durable", "lightweight", "quality", "fantastic", "brilliant", "outstanding", "impressive", "beautiful", "recommend", "reliable", "smooth", "comfortable", "delightful", "pleased", "worth", "sturdy", "elegant"},
        "negative": {"terrible", "worst", "waste", "disappointed", "regret", "poor", "uncomfortable", "horrible", "cheap", "garbage", "broken", "bad", "useless", "defect", "fail", "slow", "hate", "awful", "pathetic", "disgusting", "annoying", "frustrating", "rubbish", "mediocre", "boring", "ugly", "overpriced", "scam", "fraud", "flimsy", "defective"}
    },
    "ur": {
        "positive": {"شاندار", "بہترین", "پسند", "کمال", "اچھا", "خوبصورت", "لاجواب", "اطمینان", "سپیڈ", "پائیدار", "فائدہ", "عمدہ", "زبردست", "خوشی", "مضبوط", "بھروسہ"},
        "negative": {"ناقص", "خراب", "ضایع", "مایوسی", "افسوس", "بیکار", "فالتو", "فضول", "ٹوٹی", "کچرا", "نقصان", "برا", "نہیں", "گھٹیا", "بکواس", "مہنگا", "دھوکا"}
    },
    "ur_roman": {
        "positive": {"achi", "acha", "achchha", "acchi", "zabardast", "kamal", "amazing", "love", "best", "satisfied", "nice", "premium", "pasand", "pyari", "pyara", "speedy", "khush", "recommend", "recomended", "shandar", "behtareen", "lajawab", "mazboot", "umda", "shandaar", "khubsurat", "badhiya", "mast", "kamaal", "great", "good", "excellent", "perfect", "superb", "awesome", "wonderful"},
        "negative": {"fazool", "waste", "regret", "poor", "cheap", "worst", "garbage", "broken", "bekar", "bekaar", "kharab", "bad", "useless", "slow", "nahi", "nhi", "afsos", "fuzool", "bura", "buri", "ghatia", "ghatiya", "bakwas", "bakwaas", "faltu", "faltoo", "wahiyat", "mehnga", "mehenga", "dhoka", "tuta", "toota", "fizada", "mayoosi", "ganda", "terrible", "horrible", "awful", "hate", "disappointed", "disgusting", "pathetic", "ugly", "defective", "scam"}
    }
}

class InferencePipeline:
    """Handles end-to-end inference for review sentiment analysis."""
    def __init__(self, model_dir: str = str(FINE_TUNED_MODEL_DIR)):
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        self.model = None
        self.is_fallback_mode = False
        self.base_model_loaded = False
        
        # Check if running in Render cloud low-memory environment
        self.is_render = os.environ.get("RENDER", "false").lower() == "true"
        
        if self.is_render:
            logger.info("Running in Render Cloud Environment. Active Lexicon Heuristics bypass to respect 512MB RAM.")
            self.device = "cpu"
            self.is_fallback_mode = True
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.load_model()
            except ImportError:
                logger.info("PyTorch not installed. Starting in Lexicon Heuristics mode.")
                self.device = "cpu"
                self.is_fallback_mode = True

    def load_model(self):
        """Loads the fine-tuned model, falling back to base model or lexicon heuristic if needed."""
        # Check if fine-tuned model path exists with a config file
        config_file = self.model_dir / "config.json"
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            logger.info("Transformers not installed. Starting in Lexicon Heuristics mode.")
            self.tokenizer = None
            self.model = None
            self.is_fallback_mode = True
            return
            
        if self.model_dir.exists() and config_file.exists():
            try:
                logger.info(f"Loading fine-tuned model from {self.model_dir}...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
                self.model.to(self.device)
                self.model.eval()
                self.is_fallback_mode = False
                logger.info("Fine-tuned model loaded successfully.")
                return
            except Exception as e:
                logger.error(f"Failed to load fine-tuned model: {e}. Trying base model...")
                
        # If fine-tuned model not found, try base model from local cache only
        try:
            logger.info(f"Fine-tuned model not found. Attempting to load base model '{MODEL_NAME}' from cache...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
            self.is_fallback_mode = True
            self.base_model_loaded = True
            logger.info("Loaded base model from local cache. Using lexicon predictions with ML keyword support.")
        except Exception as e:
            logger.info("Base model not cached locally. Starting API instantly in Lexicon Heuristics mode.")
            self.tokenizer = None
            self.model = None
            self.is_fallback_mode = True

    def _lexicon_predict(self, text: str, lang: str) -> tuple[str, float]:
        """Performs dictionary-based rule-based classification as a fallback or keyword highlighter help."""
        words = get_cleaned_words(text, lang)
        
        if lang not in LEXICONS:
            # default to english if not found
            lang = "en"
            
        pos_words = LEXICONS[lang]["positive"]
        neg_words = LEXICONS[lang]["negative"]
        
        negations = {"not", "no", "dont", "doesnt", "wasnt", "isnt", "arent", "cannot", "cant", "never", "neither", "nahi", "nhi", "na", "mat", "نہ", "نہیں", "مت"}
        
        pos_score = 0
        neg_score = 0
        
        for i, word in enumerate(words):
            is_negated = False
            # Check window around the word (up to 2 words before or 2 words after)
            start_win = max(0, i - 2)
            end_win = min(len(words), i + 3)
            for j in range(start_win, end_win):
                if j != i and words[j] in negations:
                    is_negated = True
                    break
                    
            if word in pos_words:
                if is_negated:
                    neg_score += 1
                else:
                    pos_score += 1
            elif word in neg_words:
                if is_negated:
                    pos_score += 1
                else:
                    neg_score += 1
        
        # Determine sentiment
        if pos_score > neg_score:
            sentiment = "Positive"
            diff = pos_score - neg_score
            # Map diff to confidence score
            confidence = min(0.65 + (diff * 0.1), 0.95)
        elif neg_score > pos_score:
            sentiment = "Negative"
            diff = neg_score - pos_score
            confidence = min(0.65 + (diff * 0.1), 0.95)
        else:
            sentiment = "Neutral"
            # check if there are no words at all
            if len(words) == 0:
                confidence = 0.50
            else:
                confidence = 0.60
                
        return sentiment, confidence

    def predict(self, text: str) -> dict:
        """
        Analyzes a single review text.
        Returns a dictionary with sentiment, confidence, detected language, 
        prediction time, and HTML highlighted text.
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return {
                "sentiment": "Neutral",
                "confidence": 0.50,
                "language": "unknown",
                "prediction_time": 0.0,
                "highlighted_text": ""
            }
            
        # 1. Language detection
        lang = detect_language(text)
        
        # 2. Text preprocessing
        cleaned_text = preprocess_review(text, lang)
        
        # 3. Model classification or lexicon fallback
        lex_sentiment, lex_confidence = self._lexicon_predict(text, lang)
        
        # Check if lexicon negation was triggered
        words = get_cleaned_words(text, lang)
        negations = {"not", "no", "dont", "doesnt", "wasnt", "isnt", "arent", "cannot", "cant", "never", "neither", "nahi", "nhi", "na", "mat", "نہ", "نہیں", "مت"}
        has_negation = any(w in negations for w in words)
        
        if self.is_fallback_mode or self.model is None or self.tokenizer is None:
            sentiment, confidence = lex_sentiment, lex_confidence
        else:
            try:
                import torch
                # Tokenize text
                inputs = self.tokenizer(
                    cleaned_text,
                    add_special_tokens=True,
                    max_length=MAX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                pred_class_id = int(np.argmax(probs))
                
                sentiment = SENTIMENT_LABELS[pred_class_id]
                confidence = float(probs[pred_class_id])
                
                # Hybrid override: if lexicon has a negation-flipped prediction and model differs,
                # override model's output to respect negation constraints.
                if has_negation and lex_sentiment != sentiment and lex_sentiment in ["Positive", "Negative"]:
                    logger.info(f"Negation override triggered: overriding model's '{sentiment}' with lexicon '{lex_sentiment}'")
                    sentiment = lex_sentiment
                    confidence = max(confidence, lex_confidence)
                    
            except Exception as e:
                logger.error(f"Model prediction failed: {e}. Falling back to lexicon.")
                sentiment, confidence = lex_sentiment, lex_confidence
                
        execution_time = time.time() - start_time
        
        # 4. Generate highlighted text
        highlighted_html = self.highlight_keywords(text, lang)
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "language": lang,
            "prediction_time": round(execution_time, 4),
            "highlighted_text": highlighted_html,
            "using_ml_model": not (self.is_fallback_mode or self.model is None) or self.base_model_loaded
        }

    def highlight_keywords(self, text: str, lang: str) -> str:
        """Returns HTML marked string where positive and negative keywords are highlighted."""
        if lang not in LEXICONS:
            lang = "en"
            
        pos_words = LEXICONS[lang]["positive"]
        neg_words = LEXICONS[lang]["negative"]
        
        # Clean text preserving punctuation for visual markup
        # Split text into tokens and keep delimiters (like spaces, punctuation)
        tokens = re.split(r"(\s+|[.,!?;:\"()۔])", text)
        
        highlighted_tokens = []
        for token in tokens:
            # Check if token is whitespace or punctuation
            if not token or re.match(r"^(\s+|[.,!?;:\"()۔]+)$", token):
                highlighted_tokens.append(token)
                continue
                
            # Strip and lowercase token for matching
            cleaned_token = re.sub(r"[^\w]", "", token).lower()
            
            # Simple stemming helper for matching plurals/variations
            # e.g., "amazing!" -> "amazing", "products" -> "product"
            matched = False
            for w in pos_words:
                if cleaned_token == w or (len(cleaned_token) > 4 and w.startswith(cleaned_token[:-1])):
                    highlighted_tokens.append(f'<span style="background-color:rgba(46, 204, 113, 0.3); border-radius: 4px; padding: 2px 4px; border: 1px solid rgba(46, 204, 113, 0.6); font-weight: 500; color: #2ecc71;">{token}</span>')
                    matched = True
                    break
            if matched:
                continue
                
            for w in neg_words:
                if cleaned_token == w or (len(cleaned_token) > 4 and w.startswith(cleaned_token[:-1])):
                    highlighted_tokens.append(f'<span style="background-color:rgba(231, 76, 60, 0.3); border-radius: 4px; padding: 2px 4px; border: 1px solid rgba(231, 76, 60, 0.6); font-weight: 500; color: #e74c3c;">{token}</span>')
                    matched = True
                    break
            if matched:
                continue
                
            highlighted_tokens.append(token)
            
        return "".join(highlighted_tokens)
