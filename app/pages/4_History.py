import streamlit as st
import pandas as pd
from database.db_manager import DBManager
from utils.report_generator import generate_csv_report, generate_pdf_report
from pathlib import Path

# Helper to load and apply custom CSS
def apply_custom_css():
    css_path = Path(__file__).resolve().parents[1] / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

apply_custom_css()

@st.cache_resource
def get_db():
    return DBManager()

db = get_db()

st.title("📜 Prediction History")
st.markdown("Search, filter, and export the history of review predictions processed by the platform.")

# Filters Row
st.markdown("### Search & Filters")
filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])

with filter_col1:
    search_query = st.text_input("Search Reviews", placeholder="Type keywords in reviews...")
    
with filter_col2:
    sentiment_filter = st.selectbox(
        "Filter by Sentiment",
        options=["All", "Positive", "Neutral", "Negative"]
    )
    
with filter_col3:
    language_filter = st.selectbox(
        "Filter by Language",
        options=["All", "English", "Urdu", "Roman Urdu", "Hindi"]
    )

# Language value mappings for DB queries
lang_query_map = {
    "All": "All",
    "English": "en",
    "Urdu": "ur",
    "Roman Urdu": "ur_roman",
    "Hindi": "hi"
}

db_lang_filter = lang_query_map[language_filter]

# Fetch data based on filters
df_history = db.get_all_predictions(
    search_query=search_query,
    sentiment_filter=sentiment_filter,
    language_filter=db_lang_filter
)

if df_history.empty:
    st.info("No records found matching the search/filter criteria. Try broadening your criteria or run some predictions.")
else:
    st.markdown(f"Found **{len(df_history)}** records.")
    
    # Capitalize language codes for display
    df_display = df_history.copy()
    df_display["language"] = df_display["language"].str.upper()
    
    # Table layout
    st.dataframe(
        df_display[["timestamp", "review", "language", "prediction", "confidence"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Reports & Exports Actions Row
    st.markdown("### Export Reports")
    export_col1, export_col2 = st.columns(2)
    
    # Calculate statistics based on currently filtered dataframe
    filtered_total = len(df_history)
    pos_count = len(df_history[df_history["prediction"] == "Positive"])
    neg_count = len(df_history[df_history["prediction"] == "Negative"])
    neu_count = len(df_history[df_history["prediction"] == "Neutral"])
    avg_conf = df_history["confidence"].mean() if filtered_total > 0 else 0.0
    
    stats_dict = {
        "total": filtered_total,
        "avg_confidence": avg_conf,
        "positive_ratio": pos_count / filtered_total if filtered_total > 0 else 0.0,
        "negative_ratio": neg_count / filtered_total if filtered_total > 0 else 0.0,
        "neutral_ratio": neu_count / filtered_total if filtered_total > 0 else 0.0
    }
    
    with export_col1:
        # Generate CSV bytes
        csv_data = generate_csv_report(df_history)
        st.download_button(
            label="📥 Export Current View to CSV",
            data=csv_data,
            file_name="sentiment_predictions_history.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with export_col2:
        # Generate PDF bytes
        pdf_data = generate_pdf_report(df_history, stats_dict)
        st.download_button(
            label="📄 Download Executive PDF Report",
            data=pdf_data,
            file_name="sentiment_predictions_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
    st.divider()
    
    # Database Actions
    st.markdown("### Danger Zone")
    clear_confirm = st.checkbox("I confirm that I want to clear all history records from the database permanently.")
    clear_btn = st.button("Clear Prediction History", disabled=not clear_confirm, type="primary")
    
    if clear_btn:
        if db.clear_history():
            st.success("✓ Database successfully cleared.")
            st.rerun()
        else:
            st.error("Failed to clear database logs. Check logs for details.")
