import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent workspace directory to python path to resolve modules correctly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import io
import json

from database.db_manager import DBManager
from utils.inference import InferencePipeline
from utils.report_generator import generate_csv_report, generate_pdf_report
from config import FINE_TUNED_MODEL_DIR, EVAL_DIR, get_logger

logger = get_logger("api_backend")

app = FastAPI(
    title="Smart Multilingual Sentiment Analysis API",
    description="REST API backend for analyzing reviews in English, Urdu, Roman Urdu, and Hindi.",
    version="1.0.0"
)

# Enable CORS for frontend local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate singletons
db = DBManager()
pipeline = InferencePipeline()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Multilingual Sentiment Analysis API is running."}

@app.get("/api/stats")
def get_stats():
    """Retrieves overall prediction statistics from the database."""
    try:
        stats = db.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to load statistics.")

@app.post("/api/analyze")
async def analyze_review(payload: dict):
    """Analyzes sentiment of a single review text."""
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
        
    try:
        result = pipeline.predict(text)
        
        # Save to database
        db.insert_prediction(
            review=text,
            language=result["language"],
            prediction=result["sentiment"],
            confidence=result["confidence"]
        )
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing review: {e}")
        raise HTTPException(status_code=500, detail="Error executing prediction.")

@app.post("/api/bulk-upload")
async def bulk_upload(file: UploadFile = File(...), text_column: str = Form(...)):
    """Uploads a CSV file and performs batch sentiment classification."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{text_column}' not found in CSV.")
            
        predictions = []
        confidences = []
        languages = []
        prediction_times = []
        
        # Process reviews
        for _, row in df.iterrows():
            val = row[text_column]
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
                
        # Append predictions to dataframe
        df_results = df.copy()
        df_results["predicted_sentiment"] = predictions
        df_results["confidence_score"] = confidences
        df_results["detected_language"] = languages
        df_results["processing_latency_sec"] = prediction_times
        
        # Convert df to JSON records to return to frontend
        records = df_results.head(100).to_dict(orient="records") # Limit to first 100 rows for response payload size
        
        # Prepare response statistics
        total = len(df)
        pos = predictions.count("Positive")
        neg = predictions.count("Negative")
        neu = predictions.count("Neutral")
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Generate result CSV bytes to return as data URI or hold in a temporary session
        # For simplicity, we just return statistics and the preview records
        return {
            "total_rows": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "avg_confidence": round(avg_conf, 4),
            "preview_records": records
        }
        
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

@app.get("/api/history")
def get_history(
    search: str = Query(None),
    sentiment: str = Query(None),
    language: str = Query(None)
):
    """Retrieves prediction history based on filters."""
    try:
        # Convert language filter representation
        lang_map = {
            "English": "en",
            "Urdu": "ur",
            "Roman Urdu": "ur_roman"
        }
        db_lang = lang_map.get(language, language) # fallback
        
        df = db.get_all_predictions(
            search_query=search,
            sentiment_filter=sentiment,
            language_filter=db_lang
        )
        
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history.")

@app.post("/api/history/clear")
def clear_history():
    """Wipes the database logs."""
    try:
        success = db.clear_history()
        return {"success": success}
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear database.")

@app.get("/api/history/export-csv")
def export_csv():
    """Returns predictions history as a downloadable CSV stream."""
    try:
        df = db.get_all_predictions()
        csv_bytes = generate_csv_report(df)
        
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=sentiment_predictions_history.csv"}
        )
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail="Failed to compile CSV report.")

@app.get("/api/history/export-pdf")
def export_pdf():
    """Returns prediction history as a downloadable PDF stream."""
    try:
        df = db.get_all_predictions()
        stats = db.get_statistics()
        pdf_bytes = generate_pdf_report(df, stats)
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=sentiment_predictions_report.pdf"}
        )
    except Exception as e:
        logger.error(f"Error exporting PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to compile PDF report.")

@app.post("/api/load-demo")
def load_demo_data():
    """Injects mock review records into SQLite for presentation/testing."""
    try:
        db.clear_history()
        
        demo_records = [
            ("This camera quality is absolutely mind-blowing!", "en", "Positive", 0.96, datetime.now() - timedelta(days=4, hours=2)),
            ("Decent product. The display is good, but battery life is just okay.", "en", "Neutral", 0.65, datetime.now() - timedelta(days=4, hours=5)),
            ("Worst product ever! It broke within 2 hours of unboxing.", "en", "Negative", 0.98, datetime.now() - timedelta(days=3, hours=1)),
            ("Highly recommend this to everyone. Great price.", "en", "Positive", 0.94, datetime.now() - timedelta(days=2, hours=8)),
            ("Standard phone. Nothing unique but fits daily needs.", "en", "Neutral", 0.58, datetime.now() - timedelta(days=1, hours=4)),
            ("Very unhappy. The charger was missing from the box.", "en", "Negative", 0.91, datetime.now() - timedelta(hours=6)),
            
            ("یہ موبائل فون سچ میں کمال کا ہے، بہت پسند آیا۔", "ur", "Positive", 0.98, datetime.now() - timedelta(days=4, hours=6)),
            ("ٹھیک پروڈکٹ ہے، لیکن کوالٹی اور بہتر ہو سکتی تھی۔", "ur", "Neutral", 0.62, datetime.now() - timedelta(days=3, hours=4)),
            ("بالکل فضول چیز ہے، پیسے ضائع مت کریں۔", "ur", "Negative", 0.97, datetime.now() - timedelta(days=2, hours=10)),
            ("قیمت کے حساب سے بہت اچھا ہے۔", "ur", "Positive", 0.92, datetime.now() - timedelta(days=1, hours=12)),
            ("کیمرہ ٹھیک ہے لیکن بیٹری جلدی ختم ہوتی ہے۔", "ur", "Neutral", 0.55, datetime.now() - timedelta(hours=14)),
            
            ("Ye phone boht hi behtareen hai! Camera performance outstanding hai.", "ur_roman", "Positive", 0.95, datetime.now() - timedelta(days=4, hours=1)),
            ("Performance thori slow hai par use kiya ja sakta hai.", "ur_roman", "Neutral", 0.60, datetime.now() - timedelta(days=3, hours=8)),
            ("Bohat hi fazool quality hai, hang ho jata hai phone.", "ur_roman", "Negative", 0.94, datetime.now() - timedelta(days=2, hours=3)),
            ("Delivery boht fast thi aur mobile box bilkul packing me tha.", "ur_roman", "Positive", 0.91, datetime.now() - timedelta(days=1, hours=2)),
            ("Average screen, built quality normal hi hai.", "ur_roman", "Neutral", 0.57, datetime.now() - timedelta(hours=2)),
            ("Bht ganda experience raha, return apply kar diya.", "ur_roman", "Negative", 0.89, datetime.now() - timedelta(hours=12))
        ]
        
        for rev, lang, pred, conf, dt in demo_records:
            db.insert_prediction(rev, lang, pred, conf, dt)
            
        return {"status": "success", "inserted": len(demo_records)}
    except Exception as e:
        logger.error(f"Error loading demo data: {e}")
        raise HTTPException(status_code=500, detail="Failed to load demo data.")

@app.get("/api/model-details")
def get_model_details():
    """Retrieves model info and training logs status."""
    try:
        metadata_path = FINE_TUNED_MODEL_DIR / "metadata.json"
        has_trained = metadata_path.exists()
        
        meta = {}
        if has_trained:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                
        # Check if loss graphs exist
        has_loss_curve = (EVAL_DIR / "loss_curve.png").exists()
        has_acc_curve = (EVAL_DIR / "accuracy_curve.png").exists()
        has_cm = (EVAL_DIR / "confusion_matrix.png").exists()
        
        return {
            "has_trained_weights": has_trained,
            "metadata": meta,
            "has_charts": {
                "loss": has_loss_curve,
                "accuracy": has_acc_curve,
                "confusion_matrix": has_cm
            }
        }
    except Exception as e:
        logger.error(f"Error fetching model details: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model details.")
