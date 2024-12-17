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

class Config:
    """Configuration handler"""
    DEFAULT_CONFIG = {
        'processing': {
            'frame_skip': 2,  # Process every nth frame
            'batch_size': 1,  # Number of frames to process in parallel
            'detection_threshold': 0.5,  # Minimum confidence threshold
            'max_queue_size': 100,  # Maximum frames to queue for processing
        },
        'database': {
            'path': 'license_plates.db',
            'table_name': 'detections'
        },
        'output': {
            'save_video': True,
            'save_json': True,
            'display_video': True,
            'json_path': 'detections/',
            'video_path': 'processed_videos/'
        }
    }

    @staticmethod
    def load(config_path='config.yml'):
        """Load configuration from YAML file or create default"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Merge with defaults for any missing values
            return Config._merge_configs(Config.DEFAULT_CONFIG, config)
        except FileNotFoundError:
            # Save default config and return it
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(Config.DEFAULT_CONFIG, f)
            return Config.DEFAULT_CONFIG

    @staticmethod
    def _merge_configs(default, custom):
        """Recursively merge custom config with default config"""
        merged = default.copy()
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = Config._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

class DatabaseHandler:
    """Handle database operations"""
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name
        self._init_db()

    def _init_db(self):
        """Initialize database and create table if it doesn't exist"""
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
        """Store detection results in database"""
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
        # Load configuration
        self.config = Config.load(config_path)
        
        # Initialize the Nomeroff-Net pipeline
        self.number_plate_detection_pipeline = pipeline("number_plate_detection_and_reading", 
                                                      image_loader="opencv")
        
        # Initialize database handler
        self.db = DatabaseHandler(
            self.config['database']['path'],
            self.config['database']['table_name']
        )
        
        # Initialize queues for thread communication
        self.frame_queue = queue.Queue(maxsize=self.config['processing']['max_queue_size'])
        self.result_queue = queue.Queue()
        
        # Create output directories
        Path(self.config['output']['json_path']).mkdir(parents=True, exist_ok=True)
        Path(self.config['output']['video_path']).mkdir(parents=True, exist_ok=True)

    def process_frame(self, frame):
        """Process a single frame and detect license plates"""
        try:
            # Run the detection pipeline on the frame
            results = self.number_plate_detection_pipeline(frame)
            
            # Process results
            detected_plates = []
            for result in results:
                # Filter by confidence threshold
                if result.get('confidence', 0) < self.config['processing']['detection_threshold']:
                    continue
                
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, result.get('bbox', [0, 0, 0, 0]))
                
                plate_data = {
                    'text': result.get('text', ''),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(result.get('confidence', 0)),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Draw rectangle around plate
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Put text above the rectangle
                cv2.putText(frame, 
                           f"{plate_data['text']} ({plate_data['confidence']:.2f})",
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.9,
                           (0, 255, 0),
                           2)
                
                detected_plates.append(plate_data)
            
            return frame, detected_plates
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame, []

    def frame_producer(self, cap, stop_event):
        """Thread function to read frames from video source"""
        frame_count = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames according to config
            if frame_count % self.config['processing']['frame_skip'] == 0:
                try:
                    self.frame_queue.put((frame_count, frame), timeout=1)
                except queue.Full:
                    print("Frame queue is full, skipping frame")
            
            frame_count += 1
        
        # Signal end of video
        self.frame_queue.put(None)

    def frame_consumer(self, stop_event):
        """Thread function to process frames"""
        while not stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=1)
                if item is None:
                    break
                    
                frame_count, frame = item
                processed_frame, detections = self.process_frame(frame)
                self.result_queue.put((frame_count, processed_frame, detections))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame consumer: {str(e)}")
                continue
        
        # Signal end of processing
        self.result_queue.put(None)

    def process_video(self, source, output_path=None):
        """Process video file or stream and detect license plates"""
        # Open video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError("Error opening video source")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output is enabled
        writer = None
        if self.config['output']['save_video'] and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize threading events and start threads
        stop_event = threading.Event()
        producer_thread = threading.Thread(
            target=self.frame_producer,
            args=(cap, stop_event)
        )
        consumer_thread = threading.Thread(
            target=self.frame_consumer,
            args=(stop_event,)
        )
        
        producer_thread.start()
        consumer_thread.start()
        
        all_detections = []
        try:
            while True:
                result = self.result_queue.get()
                if result is None:
                    break
                
                frame_count, processed_frame, detections = result
                
                # Save detections
                if detections:
                    detection_data = {
                        'frame_number': frame_count,
                        'detections': detections
                    }
                    all_detections.append(detection_data)
                    
                    # Store in database
                    self.db.store_detections(detections, frame_count, str(source))
                
                # Write frame if enabled
                if writer:
                    writer.write(processed_frame)
                
                # Display frame if enabled
                if self.config['output']['display_video']:
                    cv2.imshow('License Plate Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            # Clean up
            stop_event.set()
            producer_thread.join()
            consumer_thread.join()
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Save detection data if enabled
            if self.config['output']['save_json'] and all_detections:
                json_filename = Path(self.config['output']['json_path']) / f'detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                with open(json_filename, 'w') as f:
                    json.dump(all_detections, f, indent=4)
                
        return all_detections

def main():
    # Example usage
    detector = LicensePlateDetector()
    
    # Process video file
    video_source = "/home/aniix/alprs/trafficvid/Car-Traffic.mp4"  # For video file
    # video_source = 0  # For webcam
    output_path = Path(detector.config['output']['video_path']) / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    try:
        detections = detector.process_video(
            source=video_source,
            output_path=output_path
        )
        print(f"Detection complete. Found {len(detections)} frames with license plates.")
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()