# part 1:

### START OF PART 1 ###
# File: license_plate_detector.py

import os
os.environ["QT_LOGGING_RULES"] = "*=false"
os.environ["QT_DEBUG_PLUGINS"] = "0"
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
from pathlib import Path
import sqlite3
from datetime import datetime
import argparse
import time
import sys
from tqdm import tqdm

class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.processing_times = []
        self.detection_count = 0
        self.frames_processed = 0
        self.start_time = time.time()
        self.active_detections = 0
        
    def update(self, process_time, num_detections=0, active_detections=0):
        self.processing_times.append(process_time)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
        self.detection_count += num_detections
        self.frames_processed += 1
        self.active_detections = active_detections

    def get_fps(self):
        if not self.processing_times:
            return 0
        return 1.0 / np.mean(self.processing_times)

    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        return {
            'fps': self.get_fps(),
            'avg_process_time': np.mean(self.processing_times) if self.processing_times else 0,
            'total_detections': self.detection_count,
            'active_detections': self.active_detections,
            'detection_rate': self.detection_count / self.frames_processed if self.frames_processed > 0 else 0,
            'elapsed_time': elapsed_time,
            'frames_processed': self.frames_processed
        }

class LicensePlateDB:
    def __init__(self, db_path='license_plates.db'):
        self.db_path = db_path
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.create_table()
    
    def create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    plate_text TEXT,
                    confidence REAL,
                    x1 INTEGER,
                    y1 INTEGER,
                    x2 INTEGER,
                    y2 INTEGER,
                    image_path TEXT,
                    source_type TEXT,
                    persistence_count INTEGER
                )
            ''')
            conn.commit()
    
    def store_detection(self, plate_text, confidence, bbox, image_path, source_type, persistence_count=1):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO detections 
                (timestamp, plate_text, confidence, x1, y1, x2, y2, image_path, source_type, persistence_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                plate_text,
                confidence,
                int(bbox[0]), int(bbox[1]),
                int(bbox[2]), int(bbox[3]),
                image_path,
                source_type,
                persistence_count
            ))
            conn.commit()

    def update_persistence(self, plate_text, persistence_count):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE detections 
                SET persistence_count = ?
                WHERE plate_text = ? AND 
                id = (SELECT id FROM detections WHERE plate_text = ? ORDER BY timestamp DESC LIMIT 1)
            ''', (persistence_count, plate_text, plate_text))
            conn.commit()

    def get_all_detections(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT *, 
                       COUNT(*) as detection_count,
                       AVG(persistence_count) as avg_persistence
                FROM detections 
                GROUP BY plate_text 
                ORDER BY timestamp DESC
            ''')
            return cursor.fetchall()

    def get_recent_detections(self, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT DISTINCT plate_text, 
                       MAX(confidence) as max_confidence,
                       COUNT(*) as detection_count,
                       MAX(persistence_count) as max_persistence
                FROM detections 
                GROUP BY plate_text 
                ORDER BY MAX(timestamp) DESC 
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()

    def get_statistics(self):
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            cursor = conn.execute('SELECT COUNT(DISTINCT plate_text) FROM detections')
            stats['unique_plates'] = cursor.fetchone()[0]
            
            cursor = conn.execute('SELECT COUNT(*) FROM detections')
            stats['total_detections'] = cursor.fetchone()[0]
            
            cursor = conn.execute('''
                SELECT source_type, COUNT(DISTINCT plate_text) 
                FROM detections 
                GROUP BY source_type
            ''')
            stats['detections_by_source'] = dict(cursor.fetchall())
            
            cursor = conn.execute('''
                SELECT AVG(confidence) as avg_confidence,
                       AVG(persistence_count) as avg_persistence
                FROM detections
            ''')
            result = cursor.fetchone()
            stats['average_confidence'] = result[0]
            stats['average_persistence'] = result[1]
            
            return stats

### END OF PART 1 ###
# Continue with Part 2, which starts with the LicensePlateDetector class