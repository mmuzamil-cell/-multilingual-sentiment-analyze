import streamlit as st
import pandas as pd
import io
import time
from database.db_manager import DBManager
from utils.inference import InferencePipeline
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
def get_pipeline():
    return InferencePipeline()

@st.cache_resource
def get_db():
    return DBManager()

pipeline = get_pipeline()
db = get_db()

st.title("📂 Bulk CSV Analysis")
st.markdown("Upload a spreadsheet (CSV) containing reviews to classify their sentiments in batches, generate statistics, and download results.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded file. Found **{len(df)}** rows and **{len(df.columns)}** columns.")
        
        # Display preview of original file
        st.markdown("### CSV Preview (First 5 Rows)")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Select review column
        columns_list = list(df.columns)
        # Try guessing the review column name
        guessed_idx = 0
        for idx, col in enumerate(columns_list):
            if col.lower() in ["review", "reviews", "text", "body", "comment", "comments"]:
                guessed_idx = idx
                break
                
        selected_col = st.selectbox(
            "Select the Column Containing Review Text",
            options=columns_list,
            index=guessed_idx
        )
        
        analyze_btn = st.button("Start Batch Analysis")
        
        if analyze_btn:
            # Check if there are rows to analyze
            if df[selected_col].isnull().all() or len(df) == 0:
                st.error("No valid text found in the selected column.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                # Setup lists to hold outputs
                predictions = []
                confidences = []
                languages = []
                prediction_times = []
                
                start_time = time.time()
                total_rows = len(df)
                
                # Perform batch predictions
                for idx, row in df.iterrows():
                    # Handle NaNs or non-string values
                    val = row[selected_col]
                    if pd.isnull(val) or not isinstance(val, str):
                        predictions.append("Neutral")
                        confidences.append(0.50)
                        languages.append("unknown")
                        prediction_times.append(0.0)
                    else:
                        res = pipeline.predict(val)
                        predictions.append(res["sentiment"])
                        confidences.append(res["confidence"])
                        languages.append(res["language"])
                        prediction_times.append(res["prediction_time"])
                        
                        # Save to database
                        db.insert_prediction(
                            review=val,
                            language=res["language"],
                            prediction=res["sentiment"],
                            confidence=res["confidence"]
                        )
                        
                    # Update progress bar
                    progress_percent = (idx + 1) / total_rows
                    progress_bar.progress(progress_percent)
                    status_text.text(f"Processing row {idx + 1} of {total_rows}...")
                    
                total_time = time.time() - start_time
                status_text.text(f"✓ Analysis complete! Processed {total_rows} rows in {total_time:.2f} seconds.")
                
                # Append prediction columns to DF
                df_results = df.copy()
                df_results["predicted_sentiment"] = predictions
                df_results["confidence_score"] = confidences
                df_results["detected_language"] = languages
                df_results["processing_latency_sec"] = prediction_times
                
                # Calculate counts
                pos_count = predictions.count("Positive")
                neu_count = predictions.count("Neutral")
                neg_count = predictions.count("Negative")
                
                # Display statistics cards
                st.markdown("### Analysis Summary")
                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                
                with sum_col1:
                    st.markdown(f"""
                    <div class='glass-card sentiment-positive' style='text-align: center;'>
                        <span style='color: #94a3b8; font-size:12px;'>POSITIVE REVIEWS</span>
                        <h2>{pos_count}</h2>
                        <span style='font-size:13px; color:#2ecc71;'>{pos_count/total_rows*100:.1f}% of total</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with sum_col2:
                    st.markdown(f"""
                    <div class='glass-card sentiment-neutral' style='text-align: center;'>
                        <span style='color: #94a3b8; font-size:12px;'>NEUTRAL REVIEWS</span>
                        <h2>{neu_count}</h2>
                        <span style='font-size:13px; color:#f39c12;'>{neu_count/total_rows*100:.1f}% of total</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with sum_col3:
                    st.markdown(f"""
                    <div class='glass-card sentiment-negative' style='text-align: center;'>
                        <span style='color: #94a3b8; font-size:12px;'>NEGATIVE REVIEWS</span>
                        <h2>{neg_count}</h2>
                        <span style='font-size:13px; color:#e74c3c;'>{neg_count/total_rows*100:.1f}% of total</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with sum_col4:
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                    st.markdown(f"""
                    <div class='glass-card' style='text-align: center;'>
                        <span style='color: #94a3b8; font-size:12px;'>AVG CONFIDENCE</span>
                        <h2>{avg_conf*100:.1f}%</h2>
                        <span style='font-size:13px; color:#00f2fe;'>Accuracy Indicator</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Download Button for Results CSV
                st.markdown("### Download Predictions")
                csv_buffer = io.BytesIO()
                df_results.to_csv(csv_buffer, index=False, encoding="utf-8")
                
                st.download_button(
                    label="📥 Download Result CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"batch_predictions_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.markdown("### Processed Data Preview")
                st.dataframe(df_results.head(10), use_container_width=True)
                
    except Exception as e:
        st.error(f"Failed to read/process file: {e}")
