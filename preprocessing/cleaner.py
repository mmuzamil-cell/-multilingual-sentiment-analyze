import re
import html
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import get_logger, LANGUAGES

logger = get_logger("cleaner")

# Initialize Lemmatizer
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    lemmatizer = WordNetLemmatizer()
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f"NLTK download failed, using fallback lists. Error: {e}")
    lemmatizer = None
    ENGLISH_STOPWORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                         "he", "him", "his", "she", "her", "it", "its", "they", "them", "their", "what", 
                         "which", "who", "whom", "this", "that", "am", "is", "are", "was", "were", "be", 
                         "been", "being", "have", "has", "had", "do", "does", "did", "but", "if", "or", 
                         "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", 
                         "against", "between", "into", "through", "during", "before", "after", "above", 
                         "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", 
                         "again", "further", "then", "once", "here", "there", "when", "where", "why", 
                         "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", 
                         "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"}

# Urdu stop words
URDU_STOPWORDS = {
    "ہے", "ہیں", "تھا", "تھی", "تھے", "گا", "گی", "گے", "کو", "نے", "سے", "کا", "کی", "کے", "میں", 
    "پر", "بھی", "تو", "ہی", "یہ", "وہ", "جو", "کر", "کیا", "کہ", "اور", "یا", "تک", "اب", "نہ", 
    "ہم", "تم", "آپ", "انہوں", "اس", "ان", "جس", "جن", "ہر", "سب", "ایک", "دو", "چند", "کچھ", "ہوں"
}

# Roman Urdu stop words
ROMAN_URDU_STOPWORDS = {
    "ye", "bohat", "acha", "hai", "ko", "se", "me", "ka", "ki", "aur", "bhi", 
    "main", "kar", "hi", "tum", "aap", "nhi", "nahi", "he", "tha", "thi", 
    "raha", "rahi", "hu", "hoon", "kuch", "ek", "par", "hota", "hoti", 
    "liya", "diya", "kam", "zayda", "zyada", "boht", "bhat", "bht", 
    "chahiye", "karne", "karna", "kya", "kiya", "thaa", "hein", "hain",
    "is", "us", "inn", "un", "wo", "woh", "ga", "ge", "gi", "to", "toh", "pe",
    "lekin", "magar", "do", "de", "chal", "mat"
}



# Try importing Urduhack and Indic NLP
try:
    import urduhack
    from urduhack.normalization import normalize as ur_normalize
    HAS_URDUHACK = True
except ImportError:
    HAS_URDUHACK = False
    logger.info("Urduhack not installed or failed to import. Using custom Urdu normalizer fallback.")




def clean_basic_text(text: str) -> str:
    """Removes HTML, URLs, emojis, and normalizes whitespaces."""
    if not text:
        return ""
    
    # Remove HTML tags
    text = html.unescape(text)
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Unicode Normalization (Form NFKC decomposes characters and normalizes compatibility characters)
    text = unicodedata.normalize('NFKC', text)
    
    # Remove extra spaces/tabs/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_urdu(text: str) -> str:
    """Normalizes Urdu characters using Urduhack or custom rules."""
    if HAS_URDUHACK:
        try:
            return ur_normalize(text)
        except Exception:
            pass
            
    # Custom Urdu Unicode character normalizations (fixing common character issues)
    # E.g. replacing Arabic kaf (ك) with Urdu kaf (ک), Arabic ya (ي) with Urdu ya, etc.
    replacements = {
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ھ": "ہ",
        "ۂ": "ہ",
        "ہہ": "ہ",
        "ة": "ہ",
        "ؤ": "و",
        "\u064B": "", # Remove Fathatan
        "\u064C": "", # Remove Dammatan
        "\u064D": "", # Remove Kasratan
        "\u064E": "", # Remove Fatha
        "\u064F": "", # Remove Damma
        "\u0650": "", # Remove Kasra
        "\u0651": "", # Remove Shadda
        "\u0652": "", # Remove Sukun
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return text





def preprocess_review(text: str, lang: str) -> str:
    """
    Performs language-specific cleaning and preprocessing on the review.
    Cleans URLs, HTML, removes stopwords, tokenizes and returns preprocessed text.
    """
    if not text:
        return ""
        
    cleaned = clean_basic_text(text)
    
    # Language specific normalizations
    if lang == "ur":
        cleaned = normalize_urdu(cleaned)

    elif lang == "en" or lang == "ur_roman":
        cleaned = cleaned.lower()
        
    return cleaned


def tokenize_text(text: str, lang: str) -> list[str]:
    """Tokenizes preprocessed text according to its language."""
    if not text:
        return []
        

            
    if lang == "ur" and HAS_URDUHACK:
        try:
            from urduhack.tokenization import sentence_tokenizer, word_tokenizer
            # basic word tokenizer
            return word_tokenizer(text)
        except Exception:
            pass
            
    # Fallback regex word tokenizer for Urdu, Roman Urdu, English
    if lang == "ur":
        # Word tokenizer that respects Urdu character clusters
        return re.findall(r"[\u0600-\u06FFa-zA-Z0-9']+", text)
    else:
        # Standard English/Roman Urdu tokenizer
        return re.findall(r"\b[a-zA-Z0-9']+\b", text)


def remove_stopwords_and_lemmatize(tokens: list[str], lang: str) -> list[str]:
    """Removes language-specific stopwords and performs lemmatization (primarily for English)."""
    if not tokens:
        return []
        
    # Choose stopword list
    if lang == "en":
        stop_words = ENGLISH_STOPWORDS
    elif lang == "ur":
        stop_words = URDU_STOPWORDS

    elif lang == "ur_roman":
        stop_words = ROMAN_URDU_STOPWORDS
    else:
        stop_words = set()
        
    # Remove stopwords
    filtered_tokens = [tok for tok in tokens if tok.lower() not in stop_words and len(tok) > 1]
    
    # Lemmatize (English only)
    if lang == "en" and lemmatizer:
        lemmatized = []
        for tok in filtered_tokens:
            try:
                # Basic noun/verb lemmatization
                lem = lemmatizer.lemmatize(tok, pos='v')
                lem = lemmatizer.lemmatize(lem, pos='n')
                lemmatized.append(lem)
            except Exception:
                lemmatized.append(tok)
        return lemmatized
        
    # No standard open-source lemmatizers for Roman Urdu, Hindi/Urdu lemmatization is complex 
    # and spaCy models are very large. Token cleanups are sufficient.
    return filtered_tokens


def get_cleaned_words(text: str, lang: str) -> list[str]:
    """Helper method to return clean list of tokens directly from raw text."""
    cleaned = preprocess_review(text, lang)
    tokens = tokenize_text(cleaned, lang)
    keywords = remove_stopwords_and_lemmatize(tokens, lang)
    return keywords
