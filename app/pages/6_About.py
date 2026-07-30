import streamlit as st
from pathlib import Path

# Helper to load and apply custom CSS
def apply_custom_css():
    css_path = Path(__file__).resolve().parents[1] / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

apply_custom_css()

st.title("ℹ️ About the Project")
st.markdown("Details regarding the project's background, methodology, stack, and NLP architectural features.")

# Methodology Card
st.markdown("## Methodology Flow")
st.markdown("""
<div class='glass-card'>
    <p style='line-height:1.6;'>
        The application is built on a <b>Deep Learning & Natural Language Processing (NLP)</b> pipeline optimized for multilingual sentiment classification. 
        It solves the challenge of understanding sentiment in Romanized Urdu (conversational Urdu written in the Latin alphabet), native Urdu (Arabic script), 
        Hindi (Devanagari script), and English.
    </p>
    <div style='margin: 20px 0;'>
        <div style='background: rgba(0, 242, 254, 0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #00f2fe;'>
            <b>1. Data Aggregation & Standardizing:</b> Reconciles multiple data sources (Amazon product reviews, Flipkart, Urdu Sentiment, and Hindi datasets) into a standard corpus format consisting of reviews, labels, language flags, and rating scores.
        </div>
        <div style='background: rgba(0, 242, 254, 0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #00f2fe; margin-top:10px;'>
            <b>2. Preprocessing & Language Routing:</b> Automatically detects the review's language. Normalizes Urdu/Hindi scripts, strips HTML/URLs, cleans double spaces, tokenizes words, and filters out language-specific stopwords.
        </div>
        <div style='background: rgba(0, 242, 254, 0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #00f2fe; margin-top:10px;'>
            <b>3. Deep Learning Fine-Tuning:</b> Fine-tunes XLM-RoBERTa Base (using a PyTorch classification head) on the merged multilingual dataset. The model's weights adapt to represent sentiment embeddings across multiple languages.
        </div>
        <div style='background: rgba(0, 242, 254, 0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #00f2fe; margin-top:10px;'>
            <b>4. Interactive Web Application:</b> Implements a Streamlit dashboard with a SQLite database backend to log review predictions, query history, run batch csv uploads, and visualize insights in real-time.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Architecture Diagram
st.markdown("## Technical Architecture")
st.markdown("""
<div class='glass-card' style='text-align: center; font-family: monospace; color:#00f2fe;'>
    <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:15px;'>
        [ User Inputs Review ] --&gt; [ Preprocessing & Unicode Normalizer ]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&[ Language Detector Router ]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[English]&nbsp;&nbsp;&nbsp;&nbsp;[Urdu]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Roman Urdu]&nbsp;&nbsp;[Hindi]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ XLM-RoBERTa Transformer Tokenizer ]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ Fine-Tuned PyTorch Classifier Classifier ]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ Output Predictions (Pos, Neu, Neg) ]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ SQLite DB logs ]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ Streamlit UI Dashboard ]
    </div>
</div>
""", unsafe_allow_html=True)

# Tech Stack Section
st.markdown("## Technical Stack & Libraries")
stack_col1, stack_col2, stack_col3 = st.columns(3)

with stack_col1:
    st.markdown("""
    <div class='glass-card' style='min-height: 180px;'>
        <h4 style='color:#00f2fe; margin-top:0;'>Backend & Core ML</h4>
        <ul style='font-size: 14px; color:#94a3b8; padding-left: 20px;'>
            <li>Python 3.12+</li>
            <li>PyTorch (Deep Learning framework)</li>
            <li>Transformers (HF Model hub)</li>
            <li>Scikit-Learn (splits & metrics)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
with stack_col2:
    st.markdown("""
    <div class='glass-card' style='min-height: 180px;'>
        <h4 style='color:#00f2fe; margin-top:0;'>Natural Language Processing</h4>
        <ul style='font-size: 14px; color:#94a3b8; padding-left: 20px;'>
            <li>NLTK (English Lemmatization)</li>
            <li>Langdetect (Language routing)</li>
            <li>Native Unicode script parsers</li>
            <li>Regex tokenization mappings</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
with stack_col3:
    st.markdown("""
    <div class='glass-card' style='min-height: 180px;'>
        <h4 style='color:#00f2fe; margin-top:0;'>Frontend & Exporters</h4>
        <ul style='font-size: 14px; color:#94a3b8; padding-left: 20px;'>
            <li>Streamlit (Web Framework)</li>
            <li>Plotly Express (Interactive charts)</li>
            <li>ReportLab (PDF report generator)</li>
            <li>WordCloud (visual keywords)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
