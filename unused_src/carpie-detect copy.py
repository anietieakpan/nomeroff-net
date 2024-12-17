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
import os
import sys

class Config:
    DEFAULT_CONFIG = {
        'processing': {
            'frame_skip': 30,          # Process every 30th frame
            'detection_threshold': 0.5,
            'max_queue_size': 2,       # Minimal queue size
            'headless': True,
            'processing_delay': 0.5,    # Longer delay between frames
            'resize_width': 640,       # Resize frames for faster processing
            'batch_processing': False   # Process one frame at a time
        },
        'database': {
            'path': 'license_plates.db',
            'table_name': 'detections'
        },
        'output': {
            'save_video': True,
            'save_json': True,
            'display_video': False,
            'json_path': 'detections/',
            'video_path': 'processed_videos/',
            'print_progress': True,
            'progress_interval': 50     # Update progress every 50 frames
        }
    }

    def __init__(self, config_path='config.yml'):
        self.config_data = self.load_config(config_path)

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return self._merge_configs(self.DEFAULT_CONFIG, config)
        except FileNotFoundError:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(self.DEFAULT_CONFIG, f)
            return self.DEFAULT_CONFIG

    def _merge_configs(self, default, custom):
        merged = default.copy()
        if custom:  # Check if custom is not None
            for key, value in custom.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_configs(merged[key], value)
                else:
                    merged[key] = value
        return merged

    def get(self, *keys):
        """Get a configuration value using dot notation"""
        value = self.config_data
        for key in keys:
            value = value[key]
        return value

class DatabaseHandler:
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    plate_text TEXT,
                    confidence REAL,
                    bbox TEXT,
                    frame_number INTEGER,
                    video_source TEXT
                )
            ''')

    def store_detections(self, detections, frame_number, video_source):
        with sqlite3.connect(self.db_path) as conn:
            for detection in detections:
                conn.execute(
                    f'INSERT INTO {self.table_name} '
                    '(timestamp, plate_text, confidence, bbox, frame_number, video_source) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        detection['timestamp'],
                        detection['text'],
                        detection['confidence'],
                        json.dumps(detection['bbox']),
                        frame_number,
                        video_source
                    )
                )

class LicensePlateDetector:
    def __init__(self, config_path='config.yml'):
        self.config = Config(config_path)
        self.detector = pipeline("number_plate_detection_and_reading")
        
        self.db = DatabaseHandler(
            self.config.get('database', 'path'),
            self.config.get('database', 'table_name')
        )
        
        self.frame_queue = queue.Queue(maxsize=self.config.get('processing', 'max_queue_size'))
        self.result_queue = queue.Queue()
        
        Path(self.config.get('output', 'json_path')).mkdir(parents=True, exist_ok=True)
        Path(self.config.get('output', 'video_path')).mkdir(parents=True, exist_ok=True)
        
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
        self.last_progress_update = 0

    def resize_frame(self, frame):
        """Resize frame while maintaining aspect ratio"""
        target_width = self.config.get('processing', 'resize_width')
        height, width = frame.shape[:2]
        if width > target_width:
            ratio = target_width / width
            new_height = int(height * ratio)
            return cv2.resize(frame, (target_width, new_height))
        return frame

    def process_frame(self, frame):
        try:
            # Resize frame for faster processing
            resized_frame = self.resize_frame(frame)
            
            # Process frame
            results = self.detector([resized_frame])
            
            detected_plates = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                
                bbox = result.get('bbox', [])
                text = result.get('text', '')
                confidence = result.get('confidence', 0.0)
                
                if not bbox or confidence < self.config.get('processing', 'detection_threshold'):
                    continue
                
                # Adjust bounding box coordinates for original frame size
                height_ratio = frame.shape[0] / resized_frame.shape[0]
                width_ratio = frame.shape[1] / resized_frame.shape[1]
                
                x1, y1, x2, y2 = map(int, bbox)
                x1, x2 = int(x1 * width_ratio), int(x2 * width_ratio)
                y1, y2 = int(y1 * height_ratio), int(y2 * height_ratio)
                
                plate_data = {
                    'text': text,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'timestamp': datetime.now().isoformat()
                }
                
                if self.config.get('output', 'save_video') or self.config.get('output', 'display_video'):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 
                              f"{text} ({confidence:.2f})",
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.9,
                              (0, 255, 0),
                              2)
                
                detected_plates.append(plate_data)
            
            return frame, detected_plates
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame, []

    def print_progress(self):
        current_time = time.time()
        if (current_time - self.last_progress_update >= 1.0 and  # Update at most once per second
            self.config.get('output', 'print_progress')):
            print(f"\rProcessed: {self.processed_frames}/{self.total_frames} frames "
                  f"(Skipped: {self.skipped_frames})", end="")
            self.last_progress_update = current_time

    def frame_producer(self, cap, stop_event):
        frame_count = 0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.config.get('processing', 'frame_skip') == 0:
                try:
                    self.frame_queue.put((frame_count, frame), timeout=0.5)
                    self.processed_frames += 1
                except queue.Full:
                    self.skipped_frames += 1
            else:
                self.skipped_frames += 1
            
            frame_count += 1
            if frame_count % self.config.get('output', 'progress_interval') == 0:
                self.print_progress()
            
            time.sleep(self.config.get('processing', 'processing_delay'))
        
        self.frame_queue.put(None)
        print("\nFinished reading frames")

    def frame_consumer(self, stop_event):
        while not stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=1)
                if item is None:
                    break
                
                frame_count, frame = item
                processed_frame, detections = self.process_frame(frame)
                self.result_queue.put((frame_count, processed_frame, detections))
                
                time.sleep(self.config.get('processing', 'processing_delay'))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nError in frame consumer: {str(e)}")
                continue
        
        self.result_queue.put(None)
        print("Finished processing frames")

    def process_video(self, source, output_path=None):
        if isinstance(source, Path):
            source = str(source)
        
        print(f"Processing video: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Error opening video source: {source}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        print(f"Video info: {width}x{height} @ {fps}fps (Expected duration: {duration:.2f}s)")
        
        writer = None
        if self.config.get('output', 'save_video') and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        stop_event = threading.Event()
        producer_thread = threading.Thread(target=self.frame_producer, args=(cap, stop_event))
        consumer_thread = threading.Thread(target=self.frame_consumer, args=(stop_event,))
        
        producer_thread.start()
        consumer_thread.start()
        
        all_detections = []
        try:
            while True:
                result = self.result_queue.get()
                if result is None:
                    break
                
                frame_count, processed_frame, detections = result
                
                if detections:
                    detection_data = {
                        'frame_number': frame_count,
                        'detections': detections
                    }
                    all_detections.append(detection_data)
                    self.db.store_detections(detections, frame_count, str(source))
                    print(f"\nFrame {frame_count}: Found plates: {[d['text'] for d in detections]}")
                
                if writer:
                    writer.write(processed_frame)
        
        except KeyboardInterrupt:
            print("\nStopping gracefully...")
        
        finally:
            print("\nCleaning up...")
            stop_event.set()
            producer_thread.join(timeout=2)
            consumer_thread.join(timeout=2)
            cap.release()
            if writer:
                writer.release()
            
            if self.config.get('output', 'save_json') and all_detections:
                json_filename = Path(self.config.get('output', 'json_path')) / f'detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                with open(json_filename, 'w') as f:
                    json.dump(all_detections, f, indent=4)
            
            print(f"\nProcessing complete:")
            print(f"Total frames: {self.total_frames}")
            print(f"Processed frames: {self.processed_frames}")
            print(f"Skipped frames: {self.skipped_frames}")
            print(f"Detected license plates: {len(all_detections)}")
                
        return all_detections

def main():
    detector = LicensePlateDetector()
    
    # Replace with your video file path
    video_source = "/home/aniix/alprs/trafficvid/Car-Traffic.mp4"  # Replace this with your actual video path
    output_path = Path(detector.config.get('output', 'video_path')) / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    try:
        print("Starting video processing...")
        print(f"Configuration:")
        print(f"- Processing every {detector.config.get('processing', 'frame_skip')}th frame")
        print(f"- Resize width: {detector.config.get('processing', 'resize_width')}px")
        print(f"- Processing delay: {detector.config.get('processing', 'processing_delay')}s")
        
        detections = detector.process_video(
            source=video_source,
            output_path=output_path
        )
        print(f"\nDetection complete. Found {len(detections)} frames with license plates.")
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()