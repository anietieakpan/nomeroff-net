import cv2
import numpy as np
from nomeroff_net import pipeline
import json
from datetime import datetime
import threading
import queue
import sqlite3
import yaml
from pathlib import Path
import time

# [Previous Config and DatabaseHandler classes remain the same]
# ... [keep all the code before LicensePlateDetector class]

class LicensePlateDetector:
    def __init__(self, config_path='config.yml'):
        self.config = Config.load(config_path)
        
        # Initialize the pipeline with correct syntax for version 4.0.0
        self.detector = pipeline("number_plate_detection_and_reading")
        
        self.db = DatabaseHandler(
            self.config['database']['path'],
            self.config['database']['table_name']
        )
        
        self.frame_queue = queue.Queue(maxsize=self.config['processing']['max_queue_size'])
        self.result_queue = queue.Queue()
        
        Path(self.config['output']['json_path']).mkdir(parents=True, exist_ok=True)
        Path(self.config['output']['video_path']).mkdir(parents=True, exist_ok=True)

# [Rest of the code remains the same]
# ... [keep all the code after LicensePlateDetector initialization]