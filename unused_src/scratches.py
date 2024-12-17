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

    def update(self, process_time, num_detections=0):
        self.processing_times.append(process_time)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
        self.detection_count += num_detections
        self.frames_processed += 1

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
            'detection_rate': self.detection_count / self.frames_processed if self.frames_processed > 0 else 0,
            'elapsed_time': elapsed_time,
            'frames_processed': self.frames_processed
        }

class LicensePlateDB:
    # [Previous DB class implementation remains the same]
    # ... [DB class code from previous version]

class LicensePlateDetector:
    def __init__(self, source_type='image', source_path=None, camera_id=0, config=None):
        self.source_type = source_type
        self.source_path = source_path
        self.camera_id = camera_id
        
        # Default configuration
        self.config = {
            'frame_skip': 4,              # Process every nth frame
            'display_scale': 0.6,         # Display window scale
            'process_scale': 1.0,         # Scale for processing
            'min_confidence': 0.5,        # Minimum detection confidence
            'show_display': True,         # Whether to show visualization
            'roi_enabled': False,         # Region of Interest processing
            'roi': None,                  # ROI coordinates [x, y, w, h]
            'save_detections': True,      # Save detections to database
            'save_video': True,           # Save processed video
            'debug_mode': False           # Show debug information
        }
        
        # Update configuration with provided values
        if config:
            self.config.update(config)

        # Initialize pipeline
        print("\nInitializing License Plate Detection System...")
        print("Loading Nomeroff-net pipeline...")
        self.pipeline = pipeline("number_plate_detection_and_reading")
        print("Pipeline loaded successfully")

        # Initialize performance monitor
        self.performance = PerformanceMonitor()
        
        # Create output directory
        Path('output').mkdir(exist_ok=True)
        
        # Initialize database if saving detections
        if self.config['save_detections']:
            self.db = LicensePlateDB()
        
        # Initialize progress bar
        self.pbar = None

    def preprocess_frame(self, frame):
        """Preprocess frame before detection"""
        # Apply ROI if enabled
        if self.config['roi_enabled'] and self.config['roi']:
            x, y, w, h = self.config['roi']
            frame = frame[y:y+h, x:x+w]
        
        # Resize frame for processing if needed
        if self.config['process_scale'] != 1.0:
            new_width = int(frame.shape[1] * self.config['process_scale'])
            new_height = int(frame.shape[0] * self.config['process_scale'])
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame

    def process_frame(self, frame):
        """Process a single frame and return visualization and detections"""
        start_time = time.time()
        
        # Preprocess frame
        process_frame = self.preprocess_frame(frame.copy())
        
        # Create visualization frame
        visualization = frame.copy()
        
        # Process with nomeroff-net
        results = self.pipeline([process_frame])
        (images, bboxs, points, zones, 
         region_ids, region_names, 
         count_lines, confidences, texts) = unzip(results)
        
        detections = []
        
        # Process detections
        if bboxs and len(bboxs[0]) > 0:
            detection_pairs = list(zip(bboxs[0], texts[0]))
            
            for bbox, text in detection_pairs:
                # Get coordinates and adjust if using ROI or processing scale
                coords = self.adjust_coordinates(bbox[:4])
                det_confidence = float(bbox[4])
                
                # Skip if confidence is too low
                if det_confidence < self.config['min_confidence']:
                    continue
                
                # Ensure text is a string
                if isinstance(text, list):
                    text = ' '.join(text)
                
                # Draw detection if display is enabled
                if self.config['show_display']:
                    self.draw_detection(visualization, coords, text, det_confidence)
                
                detections.append({
                    'text': text,
                    'confidence': det_confidence,
                    'bbox': coords
                })
        
        # Update performance metrics
        process_time = time.time() - start_time
        self.performance.update(process_time, len(detections))
        
        # Add performance info to visualization
        if self.config['show_display'] and self.config['debug_mode']:
            self.add_debug_info(visualization)
        
        return visualization, detections

    def adjust_coordinates(self, bbox):
        """Adjust coordinates based on ROI and processing scale"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Adjust for processing scale
        if self.config['process_scale'] != 1.0:
            scale = 1.0 / self.config['process_scale']
            x1, x2 = int(x1 * scale), int(x2 * scale)
            y1, y2 = int(y1 * scale), int(y2 * scale)
        
        # Adjust for ROI
        if self.config['roi_enabled'] and self.config['roi']:
            roi_x, roi_y, _, _ = self.config['roi']
            x1, x2 = x1 + roi_x, x2 + roi_x
            y1, y2 = y1 + roi_y, y2 + roi_y
        
        return [x1, y1, x2, y2]

    def draw_detection(self, frame, coords, text, confidence):
        """Draw detection box and text on frame"""
        x1, y1, x2, y2 = coords
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Prepare and draw text
        label = f"{text} ({confidence:.2f})"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        
        # Draw text background
        cv2.rectangle(frame, 
                     (x1, y1 - text_size[1] - 10),
                     (x1 + text_size[0], y1),
                     (0, 255, 0),
                     -1)
        
        # Draw text
        cv2.putText(frame, 
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    2)

    def add_debug_info(self, frame):
        """Add performance information to frame"""
        stats = self.performance.get_stats()
        info_text = [
            f"FPS: {stats['fps']:.1f}",
            f"Process Time: {stats['avg_process_time']*1000:.1f}ms",
            f"Detections: {stats['total_detections']}",
            f"Detection Rate: {stats['detection_rate']*100:.1f}%"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25

[Continued in next message...]

Would you like me to continue with the rest of the implementation, including the video processing and main function with the new optimizations?