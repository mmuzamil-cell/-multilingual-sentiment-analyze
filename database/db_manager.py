import sqlite3
import pandas as pd
from datetime import datetime
from config import DB_PATH, get_logger

logger = get_logger("db_manager")

class DBManager:
    """Manages the SQLite database for storing and querying review predictions."""
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self.init_db()

    def _get_connection(self):
        """Returns a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Initializes the database schema if it does not exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        review TEXT NOT NULL,
                        language TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp DATETIME NOT NULL
                    )
                """)
                conn.commit()
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def insert_prediction(self, review: str, language: str, prediction: str, confidence: float, timestamp: datetime = None) -> bool:
        """Inserts a new prediction record into the database."""
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO predictions (review, language, prediction, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (review, language, prediction, confidence, timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                )
                conn.commit()
            logger.info("Prediction inserted into database successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to insert prediction: {e}")
            return False

    def get_all_predictions(self, search_query: str = None, sentiment_filter: str = None, language_filter: str = None) -> pd.DataFrame:
        """Retrieves prediction history with optional filtering and search."""
        query = "SELECT id, review, language, prediction, confidence, timestamp FROM predictions WHERE 1=1"
        params = []
        
        if search_query:
            query += " AND review LIKE ?"
            params.append(f"%{search_query}%")
            
        if sentiment_filter and sentiment_filter != "All":
            query += " AND prediction = ?"
            params.append(sentiment_filter)
            
        if language_filter and language_filter != "All":
            query += " AND language = ?"
            params.append(language_filter)
            
        query += " ORDER BY timestamp DESC"
        
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Failed to retrieve predictions: {e}")
            return pd.DataFrame()

    def get_sentiment_counts(self) -> dict:
        """Returns the distribution of predicted sentiments."""
        query = "SELECT prediction, COUNT(*) as count FROM predictions GROUP BY prediction"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
            return {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.error(f"Failed to get sentiment counts: {e}")
            return {}

    def get_language_counts(self) -> dict:
        """Returns the distribution of detected languages."""
        query = "SELECT language, COUNT(*) as count FROM predictions GROUP BY language"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
            return {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.error(f"Failed to get language counts: {e}")
            return {}

    def get_predictions_over_time(self) -> pd.DataFrame:
        """Retrieves predictions aggregated by day/date for timeline plotting."""
        query = """
            SELECT date(timestamp) as date, prediction, COUNT(*) as count 
            FROM predictions 
            GROUP BY date(timestamp), prediction
            ORDER BY date(timestamp) ASC
        """
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            logger.error(f"Failed to get predictions timeline: {e}")
            return pd.DataFrame()

    def clear_history(self) -> bool:
        """Deletes all records from predictions table."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM predictions")
                conn.commit()
            logger.info("Cleared prediction history database.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False
            
    def get_statistics(self) -> dict:
        """Returns key database statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Total count
                cursor.execute("SELECT COUNT(*) FROM predictions")
                total = cursor.fetchone()[0]
                
                if total == 0:
                    return {
                        "total": 0,
                        "avg_confidence": 0.0,
                        "positive_ratio": 0.0,
                        "negative_ratio": 0.0,
                        "neutral_ratio": 0.0
                    }
                
                # Average confidence
                cursor.execute("SELECT AVG(confidence) FROM predictions")
                avg_conf = cursor.fetchone()[0] or 0.0
                
                # Sentiment breakdown
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Positive'")
                pos = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Negative'")
                neg = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Neutral'")
                neu = cursor.fetchone()[0]
                
            return {
                "total": total,
                "avg_confidence": round(avg_conf, 4),
                "positive_ratio": round(pos / total, 4),
                "negative_ratio": round(neg / total, 4),
                "neutral_ratio": round(neu / total, 4)
            }
        except Exception as e:
            logger.error(f"Failed to fetch stats: {e}")
            return {}
