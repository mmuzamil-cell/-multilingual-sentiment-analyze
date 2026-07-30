import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from database.db_manager import DBManager
from preprocessing.cleaner import get_cleaned_words
from datetime import datetime, timedelta
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

st.title("📊 Analytics Dashboard")
st.markdown("Interactive visualizations showing review sentiments, language distribution, and historic keyword trends.")

# Fetch predictions from DB
df = db.get_all_predictions()

# Handle empty database
if df.empty:
    st.warning("⚠️ No predictions found in the database. Analyze some reviews on the **Analyze Review** page or upload a CSV on **Bulk Upload** to populate the dashboard.")
    
    st.markdown("### Test Drive with Demo Data")
    st.markdown("Click the button below to pre-populate the database with 25 diverse, multilingual mock reviews spanning the last 5 days to see the analytics dashboard in action.")
    
    load_demo_btn = st.button("Load Multilingual Demo Data")
    
    if load_demo_btn:
        # Load mock predictions into the database
        demo_records = [
            # English
            ("This camera quality is absolutely mind-blowing!", "en", "Positive", 0.96, datetime.now() - timedelta(days=4, hours=2)),
            ("Decent product. The display is good, but battery life is just okay.", "en", "Neutral", 0.65, datetime.now() - timedelta(days=4, hours=5)),
            ("Worst product ever! It broke within 2 hours of unboxing.", "en", "Negative", 0.98, datetime.now() - timedelta(days=3, hours=1)),
            ("Highly recommend this to everyone. Great price.", "en", "Positive", 0.94, datetime.now() - timedelta(days=2, hours=8)),
            ("Standard phone. Nothing unique but fits daily needs.", "en", "Neutral", 0.58, datetime.now() - timedelta(days=1, hours=4)),
            ("Very unhappy. The charger was missing from the box.", "en", "Negative", 0.91, datetime.now() - timedelta(hours=6)),
            # Urdu
            ("یہ موبائل فون سچ میں کمال کا ہے، بہت پسند آیا۔", "ur", "Positive", 0.98, datetime.now() - timedelta(days=4, hours=6)),
            ("ٹھیک پروڈکٹ ہے، لیکن کوالٹی اور بہتر ہو سکتی تھی۔", "ur", "Neutral", 0.62, datetime.now() - timedelta(days=3, hours=4)),
            ("بالکل فضول چیز ہے، پیسے ضائع مت کریں۔", "ur", "Negative", 0.97, datetime.now() - timedelta(days=2, hours=10)),
            ("قیمت کے حساب سے بہت اچھا ہے۔", "ur", "Positive", 0.92, datetime.now() - timedelta(days=1, hours=12)),
            ("کیمرہ ٹھیک ہے لیکن بیٹری جلدی ختم ہوتی ہے۔", "ur", "Neutral", 0.55, datetime.now() - timedelta(hours=14)),
            # Roman Urdu
            ("Ye phone boht hi behtareen hai! Camera performance outstanding hai.", "ur_roman", "Positive", 0.95, datetime.now() - timedelta(days=4, hours=1)),
            ("Performance thori slow hai par use kiya ja sakta hai.", "ur_roman", "Neutral", 0.60, datetime.now() - timedelta(days=3, hours=8)),
            ("Bohat hi fazool quality hai, hang ho jata hai phone.", "ur_roman", "Negative", 0.94, datetime.now() - timedelta(days=2, hours=3)),
            ("Delivery boht fast thi aur mobile box bilkul packing me tha.", "ur_roman", "Positive", 0.91, datetime.now() - timedelta(days=1, hours=2)),
            ("Average screen, built quality normal hi hai.", "ur_roman", "Neutral", 0.57, datetime.now() - timedelta(hours=2)),
            ("Bht ganda experience raha, return apply kar diya.", "ur_roman", "Negative", 0.89, datetime.now() - timedelta(hours=12)),
            # Hindi
            ("यह बहुत बढ़िया उत्पाद है, कीमत भी सही है।", "hi", "Positive", 0.96, datetime.now() - timedelta(days=4, hours=11)),
            ("बैटरी साधारण चलती है, स्क्रीन की गुणवत्ता ठीक है।", "hi", "Neutral", 0.61, datetime.now() - timedelta(days=3, hours=15)),
            ("घटिया क्वालिटी है, स्क्रीन में लाइन आ गई। पैसे बर्बाद हो गए।", "hi", "Negative", 0.95, datetime.now() - timedelta(days=2, hours=18)),
            ("मुझे यह उत्पाद बहुत पसंद आया, डिलीवरी बहुत तेज़ थी।", "hi", "Positive", 0.93, datetime.now() - timedelta(days=1, hours=7)),
            ("सामान्य फ़ोन है, कोई विशेष सुविधा नहीं है।", "hi", "Neutral", 0.59, datetime.now() - timedelta(hours=18)),
            ("बिल्कुल बेकार है, कभी मत खरीदना इसे।", "hi", "Negative", 0.97, datetime.now() - timedelta(hours=4)),
        ]
        
        for rev, lang, pred, conf, dt in demo_records:
            db.insert_prediction(rev, lang, pred, conf, dt)
            
        st.success("✓ Demo data successfully generated. Reloading dashboard...")
        st.rerun()

else:
    # Get statistics
    stats = db.get_statistics()
    
    # 1. Summary Cards Row
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <span style='color: #94a3b8; font-size:12px;'>TOTAL ANALYZED</span>
            <h2>{stats.get("total", 0)}</h2>
            <span style='font-size:13px; color:#00f2fe;'>Cumulative Reviews</span>
        </div>
        """, unsafe_allow_html=True)
        
    with stat_col2:
        st.markdown(f"""
        <div class='glass-card sentiment-positive' style='text-align: center;'>
            <span style='color: #94a3b8; font-size:12px;'>POSITIVE RATIO</span>
            <h2>{stats.get("positive_ratio", 0.0) * 100:.1f}%</h2>
            <span style='font-size:13px; color:#2ecc71;'>Favorability index</span>
        </div>
        """, unsafe_allow_html=True)
        
    with stat_col3:
        st.markdown(f"""
        <div class='glass-card sentiment-negative' style='text-align: center;'>
            <span style='color: #94a3b8; font-size:12px;'>NEGATIVE RATIO</span>
            <h2>{stats.get("negative_ratio", 0.0) * 100:.1f}%</h2>
            <span style='font-size:13px; color:#e74c3c;'>Friction points</span>
        </div>
        """, unsafe_allow_html=True)
        
    with stat_col4:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <span style='color: #94a3b8; font-size:12px;'>AVG CONFIDENCE</span>
            <h2>{stats.get("avg_confidence", 0.0) * 100:.1f}%</h2>
            <span style='font-size:13px; color:#f39c12;'>Model Certainty</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("## Sentiment and Language Distribution")
    chart_col1, chart_col2 = st.columns(2)
    
    # Process plots
    # 2. Sentiment distribution pie chart
    sentiment_counts = df["prediction"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    
    # Custom colors mapping
    color_map = {
        "Positive": "#2ecc71",
        "Neutral": "#f39c12",
        "Negative": "#e74c3c"
    }
    
    with chart_col1:
        st.markdown("### Sentiment breakdown")
        fig_pie = px.pie(
            sentiment_counts, 
            values="Count", 
            names="Sentiment",
            color="Sentiment",
            color_discrete_map=color_map,
            hole=0.4,
            template="plotly_dark"
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # 3. Language distribution bar chart
    lang_counts = df["language"].value_counts().reset_index()
    lang_counts.columns = ["Language", "Count"]
    lang_counts["Language"] = lang_counts["Language"].map({
        "en": "English",
        "ur": "Urdu",
        "ur_roman": "Roman Urdu",
        "hi": "Hindi",
        "unknown": "Unknown"
    })
    
    with chart_col2:
        st.markdown("### Reviews by Language")
        fig_bar = px.bar(
            lang_counts,
            x="Language",
            y="Count",
            color="Language",
            template="plotly_dark",
            color_discrete_sequence=["#00f2fe", "#4facfe", "#203a43", "#2c5364"]
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    # 4. Predictions Timeline Chart
    st.markdown("## Prediction Timeline Trends")
    timeline_df = db.get_predictions_over_time()
    
    if not timeline_df.empty:
        # Pivot to align columns for Plotly lines
        pivot_df = timeline_df.pivot(index="date", columns="prediction", values="count").fillna(0).reset_index()
        
        # Ensure all sentiment classes exist in columns
        for sent in ["Positive", "Neutral", "Negative"]:
            if sent not in pivot_df.columns:
                pivot_df[sent] = 0.0
                
        fig_line = go.Figure(layout=go.Layout(template="plotly_dark"))
        fig_line.add_trace(go.Scatter(x=pivot_df["date"], y=pivot_df["Positive"], name="Positive", line=dict(color="#2ecc71", width=3), mode='lines+markers'))
        fig_line.add_trace(go.Scatter(x=pivot_df["date"], y=pivot_df["Neutral"], name="Neutral", line=dict(color="#f39c12", width=3), mode='lines+markers'))
        fig_line.add_trace(go.Scatter(x=pivot_df["date"], y=pivot_df["Negative"], name="Negative", line=dict(color="#e74c3c", width=3), mode='lines+markers'))
        
        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
    # 5. WordCloud Generation
    st.markdown("## Keyword Word Cloud")
    st.markdown("Extracts and compiles top sentiment-bearing words from historical queries (ignoring stop words).")
    
    # Process text for Word Cloud
    words_accumulator = []
    
    # Select language filter for word cloud
    wc_lang = st.selectbox(
        "Filter Word Cloud by Language",
        options=["All", "English", "Urdu", "Roman Urdu", "Hindi"]
    )
    
    lang_map = {
        "English": "en",
        "Urdu": "ur",
        "Roman Urdu": "ur_roman",
        "Hindi": "hi"
    }
    
    filtered_df = df
    if wc_lang != "All":
        filtered_df = df[df["language"] == lang_map[wc_lang]]
        
    for _, row in filtered_df.iterrows():
        rev = row["review"]
        lang = row["language"]
        keywords = get_cleaned_words(rev, lang)
        # Skip unknown words or very short words
        words_accumulator.extend([w for w in keywords if len(w) > 1])
        
    if not words_accumulator:
        st.info("Not enough keywords captured to render a Word Cloud. Analyze more sentences first.")
    else:
        # Create text block
        text_block = " ".join(words_accumulator)
        
        # Urdu and Hindi display in wordcloud can have font rendering issues 
        # depending on system fonts. WordCloud allows passing custom font_path.
        # We can handle fonts or fallback gracefully.
        try:
            # Create a simple word cloud
            # Deep navy background color layout
            wordcloud = WordCloud(
                width=800, 
                height=350, 
                background_color='#0b0c10', 
                colormap='GnBu',
                max_words=100
            ).generate(text_block)
            
            # Render using matplotlib
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            fig.patch.set_facecolor('#0b0c10')
            plt.tight_layout(pad=0)
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to render Word Cloud: {e}")
            
    # 6. Advanced Stats Table
    st.markdown("## Review Statistics & Breakdown")
    col_break1, col_break2 = st.columns(2)
    
    with col_break1:
        st.markdown("### Average Confidence by Sentiment")
        conf_sentiment = df.groupby("prediction")["confidence"].mean().reset_index()
        conf_sentiment.columns = ["Sentiment", "Average Confidence"]
        conf_sentiment["Average Confidence"] = conf_sentiment["Average Confidence"].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(conf_sentiment, use_container_width=True, hide_index=True)
        
    with col_break2:
        st.markdown("### Average Confidence by Language")
        conf_lang = df.groupby("language")["confidence"].mean().reset_index()
        conf_lang.columns = ["Language Code", "Average Confidence"]
        conf_lang["Language Code"] = conf_lang["Language Code"].str.upper()
        conf_lang["Average Confidence"] = conf_lang["Average Confidence"].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(conf_lang, use_container_width=True, hide_index=True)
