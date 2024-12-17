# src/storage/sqlite_storage.py
import sqlite3
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List

from .base import DetectionStorage

logger = logging.getLogger(__name__)

class SQLiteStorage(DetectionStorage):
    def __init__(self, db_path: str = 'license_plates.db'):
        self.db_path = db_path
        logger.info(f"Initializing SQLite storage at {db_path}")
        self._initialize_db()

    def _initialize_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        plate_text TEXT,
                        confidence REAL,
                        x1 INTEGER, y1 INTEGER,
                        x2 INTEGER, y2 INTEGER,
                        image_path TEXT,
                        source_type TEXT,
                        persistence_count INTEGER
                    )
                ''')
                conn.commit()
                logger.debug("Database initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def store_detection(self, plate_text: str, confidence: float,
                       bbox: List[int], image_path: str,
                       source_type: str, persistence_count: int = 1) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO detections 
                    (timestamp, plate_text, confidence, x1, y1, x2, y2, 
                     image_path, source_type, persistence_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    plate_text,
                    confidence,
                    *bbox,
                    image_path,
                    source_type,
                    persistence_count
                ))
                conn.commit()
                logger.debug(f"Stored detection: {plate_text}")
        except sqlite3.Error as e:
            logger.error(f"Failed to store detection: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                cursor = conn.execute('SELECT COUNT(DISTINCT plate_text) FROM detections')
                stats['unique_plates'] = cursor.fetchone()[0]
                
                cursor = conn.execute('''
                    SELECT AVG(confidence) as avg_confidence,
                           AVG(persistence_count) as avg_persistence
                    FROM detections
                ''')
                result = cursor.fetchone()
                stats['average_confidence'] = result[0] or 0
                stats['average_persistence'] = result[1] or 0
                
                logger.debug(f"Retrieved statistics: {stats}")
                return stats
        except sqlite3.Error as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            raise

    def cleanup(self) -> None:
        try:
            if Path(self.db_path).exists():
                Path(self.db_path).unlink()
                logger.info(f"Removed database file: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup database: {str(e)}")
            raise
