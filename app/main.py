import streamlit as st
from pathlib import Path

# Set page config at the very beginning
st.set_page_config(
    page_title="Multilingual Sentiment Analyzer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to load and apply custom CSS
def apply_custom_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Apply CSS
apply_custom_css()

# Render Sidebar metadata info
st.sidebar.markdown("""
<div style='text-align: center; padding-bottom: 20px;'>
    <h2 style='color:#00f2fe; margin-bottom: 0px;'>🔮 Smart NLP</h2>
    <p style='color:#94a3b8; font-size:12px;'>Multilingual Sentiment Analyzer</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.info("👈 Navigate through the options using the menu above to analyze reviews, upload files, or view performance statistics.")

# Home Page Content
# Modern Hero Section
st.markdown("""
<div class='glass-card' style='text-align: center; padding: 40px; background: linear-gradient(135deg, rgba(13, 27, 42, 0.4) 0%, rgba(11, 12, 16, 0.4) 100%) !important;'>
    <h1 style='font-size: 40px; margin-bottom: 10px;'>Smart Multilingual Product Review Sentiment Analysis</h1>
    <h3 style='color: #00f2fe; font-weight: 400; margin-bottom: 25px;'>Analyze Customer Feedback in English, Urdu, Roman Urdu, and Hindi using Fine-Tuned Deep Learning</h3>
    <p style='max-width: 800px; margin: 0 auto; color: #94a3b8; font-size: 16px; line-height: 1.6;'>
        Leverage the power of a fine-tuned XLM-RoBERTa (Cross-lingual Language Model - Robustly Optimized BERT Approach) 
        classifier to accurately understand reviews across multiple languages and scripts. This system detects languages 
        automatically and provides predictions of Positive, Neutral, or Negative sentiment with detailed confidence scores.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("---")

# Supported Languages Section
st.markdown("## Supported Languages")
lang_col1, lang_col2, lang_col3, lang_col4 = st.columns(4)

with lang_col1:
    st.markdown("""
    <div class='glass-card' style='text-align: center;'>
        <div style='font-size: 32px; margin-bottom: 10px;'>🇺🇸 / 🇬🇧</div>
        <h3 style='margin-bottom: 5px;'>English</h3>
        <p style='font-size:14px; color:#94a3b8; font-style:italic;'>Example: "This product is amazing."</p>
        <span class='stat-badge badge-lang'>Latin Script</span>
    </div>
    """, unsafe_allow_html=True)

with lang_col2:
    st.markdown("""
    <div class='glass-card' style='text-align: center;'>
        <div style='font-size: 32px; margin-bottom: 10px;'>🇵🇰</div>
        <h3 style='margin-bottom: 5px;'>Urdu</h3>
        <p style='font-size:14px; color:#94a3b8; font-style:italic;'>مثال: "یہ موبائل بہت اچھا ہے۔"</p>
        <span class='stat-badge badge-lang'>Arabic Script</span>
    </div>
    """, unsafe_allow_html=True)

with lang_col3:
    st.markdown("""
    <div class='glass-card' style='text-align: center;'>
        <div style='font-size: 32px; margin-bottom: 10px;'>✍️</div>
        <h3 style='margin-bottom: 5px;'>Roman Urdu</h3>
        <p style='font-size:14px; color:#94a3b8; font-style:italic;'>Example: "Ye mobile bohat acha hai."</p>
        <span class='stat-badge badge-lang'>Transliterated</span>
    </div>
    """, unsafe_allow_html=True)

with lang_col4:
    st.markdown("""
    <div class='glass-card' style='text-align: center;'>
        <div style='font-size: 32px; margin-bottom: 10px;'>🇮🇳</div>
        <h3 style='margin-bottom: 5px;'>Hindi</h3>
        <p style='font-size:14px; color:#94a3b8; font-style:italic;'>उदाहरण: "यह मोबाइल बहुत अच्छा है।"</p>
        <span class='stat-badge badge-lang'>Devanagari Script</span>
    </div>
    """, unsafe_allow_html=True)

# Features Section
st.markdown("## Core Platform Features")
feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    <div class='glass-card' style='min-height: 250px;'>
        <div style='font-size: 28px; color: #00f2fe; margin-bottom: 10px;'>🧠 Deep Learning</div>
        <h4>Fine-Tuned XLM-RoBERTa</h4>
        <p style='color: #94a3b8; font-size:14px; line-height: 1.5;'>
            Uses state-of-the-art transformer architecture to understand cross-lingual semantic relations.
            Supports complex, mixed-language reviews, and retains native language context.
        </p>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class='glass-card' style='min-height: 250px;'>
        <div style='font-size: 28px; color: #00f2fe; margin-bottom: 10px;'>⚡ Real-time Analytics</div>
        <h4>Keyword Highlighting</h4>
        <p style='color: #94a3b8; font-size:14px; line-height: 1.5;'>
            Instantly extracts and color-codes positive and negative keywords.
            Calculates confidence percentage and prediction latency in milliseconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class='glass-card' style='min-height: 250px;'>
        <div style='font-size: 28px; color: #00f2fe; margin-bottom: 10px;'>📊 Bulk Operations & Charts</div>
        <h4>Batch CSV Processing</h4>
        <p style='color: #94a3b8; font-size:14px; line-height: 1.5;'>
            Upload entire spreadsheets of customer reviews. Generate interactive charts, distribution plots, 
            word clouds, and download detailed reports as PDF or CSV.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Quick Start Guide
st.markdown("## Quick Start Guide")
st.markdown("""
<div class='glass-card'>
    <ol style='line-height: 2.0; font-size: 15px;'>
        <li>Navigate to the <b>Analyze Review</b> page in the sidebar to test predictions on individual reviews.</li>
        <li>Paste a customer review in English, Urdu, Roman Urdu, or Hindi, and click the <b>Analyze</b> button.</li>
        <li>Upload a CSV file containing user reviews on the <b>Bulk Upload</b> page to calculate sentiment distributions at scale.</li>
        <li>Browse the <b>Analytics Dashboard</b> to explore charts and see temporal prediction trends.</li>
        <li>Go to <b>History</b> to search, filter, and export past predictions to CSV/PDF reports.</li>
    </ol>
</div>
""", unsafe_allow_html=True)
