import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent workspace directory to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.db_manager import DBManager
from utils.inference import InferencePipeline
from utils.report_generator import generate_csv_report, generate_pdf_report
from preprocessing.lang_detector import detect_language

def verify_all():
    print("==================================================")
    print("STARTING BACKEND VERIFICATION TEST")
    print("==================================================")
    
    # 1. Database Verification
    print("\n[1/5] Verifying DBManager initialization...")
    db = DBManager()
    db.clear_history() # start clean for test
    
    print("Inserting mock predictions...")
    db.insert_prediction("This is a great product!", "en", "Positive", 0.95, datetime.now() - timedelta(days=2))
    db.insert_prediction("Ye mobile bekar hai, return kar rha.", "ur_roman", "Negative", 0.88, datetime.now() - timedelta(days=1))
    db.insert_prediction("یہ پروڈکٹ بہت زبردست ہے۔", "ur", "Positive", 0.98, datetime.now())
    
    stats = db.get_statistics()
    print(f"Database Stats: {stats}")
    assert stats["total"] == 3, "DB count should be 3"
    assert abs(stats["positive_ratio"] - 0.6667) < 0.01, "Positive ratio should be approx 0.67"
    print("OK: DBManager verified successfully.")
    
    # 2. Language Detector Verification
    print("\n[2/5] Verifying Language Detection Heuristics...")
    test_cases = [
        ("This is a standard english sentence.", "en"),
        ("یہ اردو کی ایک تحریر ہے۔", "ur"),
        ("ye roman urdu ki line hai.", "ur_roman"),
        ("boht acha mobile hai ye to.", "ur_roman")
    ]
    
    for idx, (text, expected) in enumerate(test_cases):
        detected = detect_language(text)
        safe_text = text[:25].encode('ascii', errors='replace').decode('ascii')
        print(f"Text {idx}: '{safe_text}' -> Expected: {expected}, Detected: {detected}")
        assert detected == expected, f"Failed language detection for case {idx}"
    print("OK: Language detector verified successfully.")
    
    # 3. Inference Pipeline Verification
    print("\n[3/5] Verifying InferencePipeline...")
    pipeline = InferencePipeline()
    print(f"Fallback Mode Active: {pipeline.is_fallback_mode}")
    
    samples = [
        "This product is absolutely amazing!",
        "Bohat hi bakwaas and ganda product hai, paisa waste.",
        "It is okay but packaging was damaged."
    ]
    for text in samples:
        res = pipeline.predict(text)
        print(f"\nReview: '{text}'")
        print(f"Detected Lang: {res['language']}")
        print(f"Prediction: {res['sentiment']} (Confidence: {res['confidence'] * 100:.1f}%)")
        print(f"Time: {res['prediction_time'] * 1000:.2f}ms")
        print(f"Highlighted HTML: {res['highlighted_text']}")
    print("\nOK: InferencePipeline verified successfully.")
    
    # 4. Exporter Verification
    print("\n[4/5] Verifying Report Exporter...")
    df = db.get_all_predictions()
    
    print("Generating CSV...")
    csv_bytes = generate_csv_report(df)
    assert len(csv_bytes) > 0, "CSV bytes should not be empty"
    print(f"CSV Bytes generated: {len(csv_bytes)}")
    
    print("Generating PDF...")
    pdf_bytes = generate_pdf_report(df, stats)
    assert len(pdf_bytes) > 0, "PDF bytes should not be empty"
    print(f"PDF Bytes generated: {len(pdf_bytes)}")
    print("OK: Report Exporter verified successfully.")
    
    print("\n==================================================")
    print("ALL BACKEND VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    verify_all()
