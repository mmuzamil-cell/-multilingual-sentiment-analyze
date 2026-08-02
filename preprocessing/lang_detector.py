import re
from langdetect import detect_langs, DetectorFactory
from config import get_logger

# Ensure reproducible results from langdetect
DetectorFactory.seed = 0

logger = get_logger("lang_detector")

# Common Roman Urdu stopwords and markers
ROMAN_URDU_KEYWORDS = {
    "ye", "bohat", "acha", "hai", "ko", "se", "me", "ka", "ki", "aur", "bhi", 
    "main", "kar", "hi", "tum", "aap", "nhi", "nahi", "he", "tha", "thi", 
    "raha", "rahi", "hu", "hoon", "kuch", "ek", "par", "hota", "hoti", 
    "liya", "diya", "kam", "zayda", "zyada", "boht", "bhat", "bht", 
    "chahiye", "karne", "karna", "kya", "kiya", "thaa", "hein", "hain",
    "parha", "likha", "kharab", "bekar", "lekin", "magar", "toh", "to",
    "pe", "is", "us", "inn", "un", "wo", "woh", "ga", "ge", "gi",
    "chal", "de", "do", "mat", "kabhi", "saath", "sath", "baad", "bad",
    "pehle", "pehlay", "pehla", "pehli", "dur", "door", "pass", "paas"
}

# Words that overlap between common English and Roman Urdu representation
CONFLICT_WORDS = {"is", "us", "to", "me", "he", "main", "pass", "bad"}
PURE_ROMAN_URDU_KEYWORDS = ROMAN_URDU_KEYWORDS - CONFLICT_WORDS

# Common English indicator words to offset short text conflicts
ENGLISH_INDICATORS = {
    "this", "the", "product", "very", "good", "great", "bad", "worst", "was", "for", "with", 
    "that", "it", "my", "you", "they", "we", "are", "about", "am", "an", "terrible", "horrible", 
    "excellent", "perfect", "satisfied", "disappointed", "love", "like", "recommend", "buying",
    "bought", "item", "purchase", "one", "too", "so", "extremely"
}

def detect_language(text: str) -> str:
    """
    Detects the language of the given review.
    Supported labels returned:
        - 'en': English
        - 'ur': Urdu (native script)
        - 'hi': Hindi (devanagari script)
        - 'ur_roman': Roman Urdu (latin script)
        - 'unknown': Other/undetected languages
    """
    if not text or not isinstance(text, str):
        return "unknown"
        
    text_clean = text.strip()
    if not text_clean:
        return "unknown"
        
    # 1. Native Urdu script detection (Arabic/Perso-Arabic script Unicode range: U+0600 to U+06FF)
    # Also include extended arabic characters for urdu (U+0750 to U+077F, U+FB50 to U+FDFF, U+FE70 to U+FEFF)
    urdu_pattern = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
    urdu_chars = urdu_pattern.findall(text_clean)
    if len(urdu_chars) > 0 and (len(urdu_chars) / len(text_clean)) > 0.15:
        return "ur"
        
    # 2. Handle Latin-based scripts: English and Roman Urdu
    # Clean text to keep alphabetic characters for dictionary matching
    words = re.findall(r"\b[a-zA-Z']+\b", text_clean.lower())
    if not words:
        return "unknown"
        
    # Calculate counts
    roman_urdu_word_count = sum(1 for w in words if w in ROMAN_URDU_KEYWORDS)
    pure_roman_urdu_count = sum(1 for w in words if w in PURE_ROMAN_URDU_KEYWORDS)
    english_count = sum(1 for w in words if w in ENGLISH_INDICATORS)
    
    roman_urdu_ratio = roman_urdu_word_count / len(words)
    
    # Refined Roman Urdu classification
    is_roman_urdu = False
    if pure_roman_urdu_count >= 1:
        # If pure Roman Urdu words are present, it's highly likely Roman Urdu
        is_roman_urdu = True
    elif roman_urdu_word_count >= 1:
        # Only overlapping conflict words matched
        # If English indicator words are dominant, it's English, otherwise if short it's Roman Urdu
        if english_count > 0 and english_count >= roman_urdu_word_count:
            is_roman_urdu = False
        elif len(words) <= 5:
            is_roman_urdu = True
            
    if is_roman_urdu:
        return "ur_roman"
        
    # 4. Fall back to langdetect for general-purpose language detection
    try:
        predictions = detect_langs(text_clean)
        top_pred = predictions[0]
        
        if top_pred.lang == "en" and top_pred.prob > 0.5:
            return "en"
        elif top_pred.lang == "ur":
            return "ur"
        
        # If it was detected as something like Italian, Romanian, or Tagalog, 
        # and has at least one Roman Urdu word, it's likely Roman Urdu.
        if roman_urdu_word_count >= 1:
            return "ur_roman"
            
        # If the top prediction is English, check if we have any strong indicators of english
        if top_pred.lang in ["en", "so", "af", "nl", "no"]: # common false positives for short text
            return "en"
            
        return "unknown"
    except Exception as e:
        logger.debug(f"langdetect failed for text: '{text_clean[:30]}...', error: {e}")
        # Final fallback
        if roman_urdu_word_count > 0:
            return "ur_roman"
        return "en" # Default to English if it is Latin-based and langdetect fails
