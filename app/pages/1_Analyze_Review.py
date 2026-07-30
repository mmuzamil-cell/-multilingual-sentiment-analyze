import streamlit as st
from utils.inference import InferencePipeline
from database.db_manager import DBManager
from pathlib import Path

# Helper to load and apply custom CSS
def apply_custom_css():
    css_path = Path(__file__).resolve().parents[1] / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

apply_custom_css()

# Instantiate pipeline and DB manager
# We cache the pipeline to avoid re-loading tokenizer/model on every rerun
@st.cache_resource
def get_pipeline():
    return InferencePipeline()

@st.cache_resource
def get_db():
    return DBManager()

pipeline = get_pipeline()
db = get_db()

# Title
st.title("🔮 Analyze Review")
st.markdown("Analyze a single product review in real time. The deep learning model will automatically detect the language and predict its sentiment.")

# Text input
st.markdown("### Enter Review Text")
review_text = st.text_area(
    label="Review Text",
    placeholder="Type a review here (e.g. 'This camera quality is fantastic!' or 'Ye mobile bohat ganda hai, mat buy karein' or 'یہ موبائل بہت شاندار ہے')...",
    height=150,
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 5])
with col1:
    analyze_btn = st.button("Analyze Sentiment", use_container_width=True)

if analyze_btn:
    if not review_text.strip():
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        # Show prediction loading state
        with st.spinner("🔮 Running deep learning classification..."):
            # Predict
            result = pipeline.predict(review_text)
            
            # Save to SQLite DB
            db.insert_prediction(
                review=review_text,
                language=result["language"],
                prediction=result["sentiment"],
                confidence=result["confidence"]
            )
            
        # Display Results
        st.markdown("### Analysis Results")
        
        # Color coding class
        sentiment = result["sentiment"]
        confidence = result["confidence"]
        lang = result["language"]
        pred_time = result["prediction_time"]
        highlighted = result["highlighted_text"]
        
        # Select theme border color class
        border_class = ""
        badge_class = ""
        text_color = ""
        
        if sentiment == "Positive":
            border_class = "sentiment-positive"
            badge_class = "badge-pos"
            text_color = "#2ecc71"
        elif sentiment == "Neutral":
            border_class = "sentiment-neutral"
            badge_class = "badge-neu"
            text_color = "#f39c12"
        else:
            border_class = "sentiment-negative"
            badge_class = "badge-neg"
            text_color = "#e74c3c"
            
        # Language display text
        lang_display = {
            "en": "English 🇺🇸/🇬🇧",
            "ur": "Urdu 🇵🇰",
            "ur_roman": "Roman Urdu ✍️",
            "hi": "Hindi 🇮🇳",
            "unknown": "Unknown ❓"
        }.get(lang, lang.upper())
        
        # HTML card template
        st.markdown(f"""
        <div class='glass-card {border_class}'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
                <div>
                    <h3 style='margin: 0px;'>Prediction Status</h3>
                    <p style='color:#94a3b8; font-size:12px; margin: 0px;'>Processed in {pred_time * 1000:.2f} ms</p>
                </div>
                <span class='stat-badge {badge_class}' style='font-size: 16px; padding: 6px 16px;'>{sentiment}</span>
            </div>
            
            <div style='margin-bottom: 20px;'>
                <h4 style='margin-bottom: 5px; color:#94a3b8; font-weight: 500; font-size: 14px;'>REVIEW WORD HIGHLIGHTS</h4>
                <div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; font-size:16px; line-height: 1.8;'>
                    {highlighted}
                </div>
            </div>
            
            <div style='display: flex; gap: 30px;'>
                <div>
                    <span style='color: #94a3b8; font-size: 12px; display: block;'>DETECTED LANGUAGE</span>
                    <span style='font-weight: 600; font-size: 15px; color: #ffffff;'>{lang_display}</span>
                </div>
                <div>
                    <span style='color: #94a3b8; font-size: 12px; display: block;'>PREDICTION CONFIDENCE</span>
                    <span style='font-weight: 600; font-size: 15px; color: {text_color};'>{confidence * 100:.2f}%</span>
                </div>
                <div>
                    <span style='color: #94a3b8; font-size: 12px; display: block;'>CLASSIFIER ENGINE</span>
                    <span style='font-weight: 600; font-size: 15px; color: #ffffff;'>{"Fine-Tuned XLM-R Model" if result.get("using_ml_model", True) else "Lexicon Helper Heuristics"}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display visual probability gauge/progress
        progress_val = int(confidence * 100)
        st.markdown(f"**Confidence Gauge**")
        st.progress(confidence)
        
        # Display a disclaimer if the model is operating in fallback/lexicon mode
        if not result.get("using_ml_model", True):
            st.warning("⚠️ **Note**: The custom fine-tuned deep learning model is still downloading or training. Lexicon Heuristics fallbacks are currently running to ensure predictable sentiment classifications. Once training is complete, the neural classifier will activate automatically.")
