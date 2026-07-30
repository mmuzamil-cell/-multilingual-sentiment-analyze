import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import DATA_DIR, get_logger, RANDOM_SEED
from preprocessing.lang_detector import detect_language

logger = get_logger("dataset_builder")

def generate_synthetic_data() -> pd.DataFrame:
    """Generates a rich, multilingual dataset for testing and training when no real datasets are provided."""
    logger.info("Generating synthetic multilingual product reviews dataset...")
    
    # Review templates to generate diverse data
    templates = {
        "en": {
            "Positive": [
                "This product is absolutely amazing, highly recommended!",
                "Great battery life and crystal clear screen. Love it!",
                "Excellent quality for the price. Fast delivery too.",
                "Really satisfied with my purchase. Will buy again.",
                "Superb build quality and very user-friendly interface.",
                "Absolutely love this item! Works like a charm.",
                "Five stars! Exceeded my expectations in every way.",
                "The customer service was great and the product is top notch.",
                "Very lightweight, portable, and extremely durable.",
                "Best purchase I have made this year. High quality."
            ],
            "Neutral": [
                "It is okay, nothing special but works fine.",
                "Decent product, build quality could be better.",
                "Average performance, average battery life.",
                "Just regular quality, does the job as advertised.",
                "Satisfactory purchase. Not bad, not great.",
                "Average experience. The product works but has some minor issues.",
                "It is fine for basic use, but professional users might want more.",
                "Shipping was fast but the product is just average.",
                "Mediocre product. It works but feels cheap.",
                "Fair value for the price, but don't expect too much."
            ],
            "Negative": [
                "Terrible product, broke on the first day of use!",
                "Worst purchase ever, absolute waste of money.",
                "Very disappointed with the quality. Highly regret buying.",
                "Battery drains in less than 2 hours. Do not buy this.",
                "Poor design, very uncomfortable to use. Returning it.",
                "Horrible experience. The item stopped working after a week.",
                "Extremely low quality. The description was completely misleading.",
                "The product looks cheap and functions even worse.",
                "Do not waste your money on this garbage.",
                "Customer support was unhelpful and the product is broken."
            ]
        },

        "ur": {
            "Positive": [
                "یہ پروڈکٹ واقعی شاندار ہے، میں اس کی انتہائی سفارش کرتا ہوں!",
                "بہترین بیٹری لائف اور شاندار ڈسپلے۔ مجھے یہ پسند آیا!",
                "قیمت کے لحاظ سے بہترین معیار۔ تیز ڈیلیوری بھی۔",
                "اپنی خریداری سے واقعی مطمئن ہوں۔ دوبارہ خریدوں گا۔",
                "شاندار بناوٹ اور استعمال میں انتہائی آسان انٹرفیس۔",
                "اس چیز سے بالکل پیار ہو گیا ہے! بہترین کام کرتی ہے۔",
                "پانچ ستارے! میری توقعات سے بڑھ کر نکلا۔",
                "سروس بہت اچھی تھی اور پروڈکٹ بھی لاجواب ہے۔",
                "بہت ہلکا پھلکا، پورٹیبل اور انتہائی پائیدار۔",
                "اس سال کی میری بہترین خریداری۔ اعلیٰ ترین معیار۔"
            ],
            "Neutral": [
                "یہ عام سی چیز ہے، نہ زیادہ اچھی نہ خراب۔ کام چل جاتا ہے۔",
                "ٹھیک ہے لیکن ڈیزائن اور کوالٹی مزید بہتر ہو سکتی تھی۔",
                "اوسط کارکردگی ہے، کوئی خاص بات نہیں۔ ٹھیک ہے۔",
                "بس عام معیار کا ہے، جیسا بتایا گیا تھا ویسا ہی ہے۔",
                "تسلی بخش خریداری۔ نہ برا ہے نہ بہت اچھا۔",
                "اوسط تجربہ رہا۔ پروڈکٹ کام کرتی ہے لیکن کچھ معمولی مسائل ہیں۔",
                "بنیادی استعمال کے لیے ٹھیک ہے، لیکن پیشہ ورانہ استعمال کے لیے نہیں۔",
                "قیمت کے لحاظ سے مناسب ہے لیکن زیادہ توقع نہ رکھیں۔",
                "کام تو کرتا ہے لیکن بناوٹ سستی معلوم ہوتی ہے۔",
                "نارمل پروڈکٹ ہے، روزمرہ کام کے لیے مناسب ہے۔"
            ],
            "Negative": [
                "بہت ہی ناقص پروڈکٹ ہے، پہلے ہی دن خراب ہو گئی!",
                "پیسے ضائع ہو گئے، بالکل بھی نہ خریدیں۔ سخت مایوسی ہوئی۔",
                "کوالٹی سے شدید مایوسی ہوئی۔ خریدنے پر افسوس ہے۔",
                "بیٹری 2 گھنٹے بھی نہیں چلتی۔ اسے بالکل مت خریدیں۔",
                "ناقص ڈیزائن، استعمال کرنے میں بہت تکلیف دہ۔ واپس کر رہا ہوں۔",
                "انتہائی خراب تجربہ۔ ایک ہفتے کے بعد چلنا بند ہو گیا۔",
                "بہت ہی گھٹیا کوالٹی۔ تفصیلات بالکل جھوٹ پر مبنی تھیں۔",
                "پروڈکٹ دکھنے میں بھی سستی ہے اور کام بھی نہیں کرتی۔",
                "اس کچرے پر اپنے پیسے برباد مت کریں۔",
                "کسٹمر سپورٹ کا رویہ خراب تھا اور پروڈکٹ ٹوٹی ہوئی ملی۔"
            ]
        },
        "ur_roman": {
            "Positive": [
                "Ye product bohat achi hai, main iski high recommendation deta hoon!",
                "Battery backup boht zabardast hai aur screen quality bhi kamal hai.",
                "Price ke hisab se behtareen quality hai. Delivery bhi fast thi.",
                "Zabardast product hai, boht pasand aya. Zaroor dobara loonga.",
                "Build quality kamaal ki hai aur use karna boht easy hai.",
                "Absolutely love this item! Bohot hi achi performance hai.",
                "Five stars! Meri umeedon se zyada acha nikla ye mobile.",
                "Customer care boht acha tha aur product to premium quality ki hai.",
                "Boht lightweight hai, easily carry ho jata hai aur solid hai.",
                "Best purchase hai meri is saal ki. Maza aa gaya use karke."
            ],
            "Neutral": [
                "Theek thaak product hai, normal kaam karta hai. Kuch special nahi.",
                "Normal quality hai, build quality thodi aur behtar ho sakti thi.",
                "Average performance hai, battery timing bhi normal hi hai.",
                "Bas regular quality hai, jesa bataya gaya tha wesa hi kaam karta hai.",
                "Satisafactory purchase hai. Na boht acha na kharab.",
                "Avg experience raha. Chal to rha hai par thode issues hain.",
                "Basic usage ke liye thik hai, high-end works ke liye suitable nahi.",
                "Price ke hisab se okay product hai, zyada umeed mat rakhein.",
                "Work karta hai par feel kafi cheap quality ki hai.",
                "Normal sa phone hai, simple requirements ke liye standard hai."
            ],
            "Negative": [
                "Bohat hi fazool product hai, pehle din hi kharab ho gyi!",
                "Paisa waste hai, bilkul mat khareedein. Boht afsos hua buy karke.",
                "Worst quality ever. Regret ho raha hai paise kharab karke.",
                "Battery 2 ghante bhi nahi chalti. Waste of money, don't buy.",
                "Poor design hai, use karne me boht mushkil ho rahi hai.",
                "Horrible experience! Ek week me hi display chala gaya iska.",
                "Extremely low quality. Description bilkul galat thi iski.",
                "Product boht sasti lagti hai aur kaam bilkul nahi karti.",
                "Is kachray pe apna paisa zaya mat karein, bekar hai.",
                "Broken item mili aur support walon ne response bhi nahi diya."
            ]
        }
    }
    
    # Generate expanded records by slightly varying reviews to create ~800 records
    data = []
    
    # Rating mappings based on sentiment
    rating_map = {
        "Positive": [4, 5],
        "Neutral": [3],
        "Negative": [1, 2]
    }
    
    sentiment_to_label = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
    
    # Generate data
    for lang, sentiments in templates.items():
        for sentiment, review_list in sentiments.items():
            for review in review_list:
                # Add original
                rating = int(np.random.choice(rating_map[sentiment]))
                data.append({
                    "review": review,
                    "label": sentiment_to_label[sentiment],
                    "language": lang,
                    "rating": rating
                })
                
                # Create 4 variations per review to expand dataset size (e.g. adding punctuation, prefixes, suffixes)
                prefixes = ["", "Aao batata hoon, ", "Honestly, ", "Mera review ye hai k ", "To be fair, ", "Mera sacha reviews: ", "سچ کہوں تو، "]
                suffixes = ["", " Highly recommended.", " worth buying.", " bad experience.", " recommended to all.", " !!!", ".", " 💯", " 👍", " 👎"]
                
                for idx in range(4):
                    prefix = ""
                    suffix = ""
                    # Pick language appropriate variation additions
                    if lang == "en":
                        prefix = np.random.choice(["Honestly, ", "Actually, ", "My honest review: ", ""])
                        suffix = np.random.choice([" Really nice.", " Disappointed.", " Recommended.", "!", "."])

                    elif lang == "ur":
                        prefix = np.random.choice(["سچ تو یہ ہے کہ، ", "میرا ذاتی تجربہ ہے کہ، ", ""])
                        suffix = np.random.choice([" بہت زبردست پروڈکٹ ہے۔", " مایوسی ہوئی۔", " شکریہ۔", "۔"])
                    elif lang == "ur_roman":
                        prefix = np.random.choice(["Mera opinion ye hai k ", "Sachi baat to ye hai k ", ""])
                        suffix = np.random.choice([" Boht pyari product hai.", " Don't buy.", " Paisa vasool.", "!!!", "."])
                        
                    var_review = f"{prefix}{review}{suffix}".strip()
                    # Clean up duplicate punctuation
                    var_review = re.sub(r'\.+', '.', var_review)
                    var_review = re.sub(r'!+', '!', var_review)
                    
                    rating = int(np.random.choice(rating_map[sentiment]))
                    data.append({
                        "review": var_review,
                        "label": sentiment_to_label[sentiment],
                        "language": lang,
                        "rating": rating
                    })
                    
    df = pd.DataFrame(data)
    # Shuffle dataset
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df

def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to load datasets from files, standardize them, or
    generate synthetic dataset. Returns train, val, test splits.
    """
    datasets_found = []
    
    # Try reading real datasets if they are stored in data/ folder
    # List of expected filenames
    file_mappings = {
        "amazon.csv": ("review", "label", "language", "rating"),
        "flipkart.csv": ("review", "label", "language", "rating"),
        "multilingual_amazon.csv": ("review", "label", "language", "rating"),
        "hindi_sentiment.csv": ("review", "label", "language", "rating"),
        "urdu_sentiment.csv": ("review", "label", "language", "rating")
    }
    
    for filename, cols in file_mappings.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            try:
                logger.info(f"Found real dataset file: {filename}. Loading...")
                df = pd.read_csv(filepath)
                # If column names differ, map them (basic mapping logic)
                # Assumes simple mappings or existence of standard columns.
                # Standardize columns: review, label, language, rating
                # If rating is missing, default to 3 for neutral, 5 for positive, 1 for negative
                # Let's perform validation:
                needed_cols = ["review", "label", "language", "rating"]
                for col in needed_cols:
                    if col not in df.columns:
                        if col == "language":
                            # Detect language programmatically
                            df["language"] = df["review"].apply(detect_language)
                        elif col == "rating":
                            # Estimate rating from label
                            # e.g., if label is 2 (positive) -> 5, 0 (negative) -> 1, 1 (neutral) -> 3
                            df["rating"] = df["label"].map({2: 5, 1: 3, 0: 1, "Positive": 5, "Neutral": 3, "Negative": 1})
                            df["rating"] = df["rating"].fillna(3)
                
                df_standard = df[["review", "label", "language", "rating"]].copy()
                # Ensure labels are 0, 1, 2
                df_standard["label"] = df_standard["label"].map({
                    "Negative": 0, "Neutral": 1, "Positive": 2,
                    0: 0, 1: 1, 2: 2,
                    "negative": 0, "neutral": 1, "positive": 2,
                    "neg": 0, "neu": 1, "pos": 2,
                    "1": 0, "2": 1, "3": 2 # support string mapping
                })
                # Drop rows with NaN labels or reviews
                df_standard = df_standard.dropna(subset=["review", "label"])
                df_standard["label"] = df_standard["label"].astype(int)
                df_standard["rating"] = df_standard["rating"].astype(int)
                datasets_found.append(df_standard)
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                
    if datasets_found:
        logger.info(f"Combining {len(datasets_found)} local datasets...")
        combined_df = pd.concat(datasets_found, ignore_index=True)
        # Drop duplicates
        combined_df = combined_df.drop_duplicates(subset=["review"])
        combined_df = combined_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        logger.info("No real datasets found. Using rich synthetic dataset generator.")
        combined_df = generate_synthetic_data()
        
    # Save combined dataset to CSV
    combined_path = DATA_DIR / "combined_standardized.csv"
    combined_df.to_csv(combined_path, index=False, encoding="utf-8")
    logger.info(f"Saved standardized dataset ({len(combined_df)} records) to {combined_path}")
    
    # Log distribution statistics
    logger.info(f"Label distribution:\n{combined_df['label'].value_counts(normalize=True)}")
    logger.info(f"Language distribution:\n{combined_df['language'].value_counts()}")
    
    # Splits (Train: 80%, Validation: 10%, Test: 10%)
    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=0.20, 
        random_state=RANDOM_SEED, 
        stratify=combined_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=RANDOM_SEED, 
        stratify=temp_df["label"]
    )
    
    # Save splits
    train_df.to_csv(DATA_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(DATA_DIR / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(DATA_DIR / "test.csv", index=False, encoding="utf-8")
    
    logger.info(f"Dataset splits generated: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    build_dataset()
