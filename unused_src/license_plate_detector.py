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

class LicensePlateDB:
    def __init__(self, db_path='license_plates.db'):
        self.db_path = db_path
        # If database exists, remove it to ensure correct schema
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
                    source_type TEXT
                )
            ''')
            conn.commit()
    
    def store_detection(self, plate_text, confidence, bbox, image_path, source_type):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO detections 
                (timestamp, plate_text, confidence, x1, y1, x2, y2, image_path, source_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                plate_text,
                confidence,
                int(bbox[0]), int(bbox[1]),
                int(bbox[2]), int(bbox[3]),
                image_path,
                source_type
            ))
            conn.commit()

    def get_all_detections(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM detections ORDER BY timestamp DESC')
            return cursor.fetchall()

    def get_recent_detections(self, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?',
                (limit,)
            )
            return cursor.fetchall()

    def get_statistics(self):
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            # Total detections
            cursor = conn.execute('SELECT COUNT(*) FROM detections')
            stats['total_detections'] = cursor.fetchone()[0]
            
            # Detections by source type
            cursor = conn.execute('''
                SELECT source_type, COUNT(*) 
                FROM detections 
                GROUP BY source_type
            ''')
            stats['detections_by_source'] = dict(cursor.fetchall())
            
            # Average confidence
            cursor = conn.execute('SELECT AVG(confidence) FROM detections')
            stats['average_confidence'] = cursor.fetchone()[0]
            
            return stats

class LicensePlateDetector:
    def __init__(self, source_type='image', source_path=None, camera_id=0):
        self.source_type = source_type
        self.source_path = source_path
        self.camera_id = camera_id
        
        # Initialize pipeline
        print("\nInitializing License Plate Detection System...")
        print("Loading Nomeroff-net pipeline...")
        self.pipeline = pipeline("number_plate_detection_and_reading")
        print("Pipeline loaded successfully")
        
        # Create output directory
        Path('output').mkdir(exist_ok=True)
        
        # Initialize database
        self.db = LicensePlateDB()
        
        # Processing settings
        self.frame_skip = 10  # Process every nth frame
        self.min_confidence = 0.5
        self.display_scale = 0.7  # 70% of original size
        
        # Performance monitoring
        self.processing_times = []
        self.max_processing_times = 100  # Keep last 100 measurements

    def process_frame(self, frame):
        """Process a single frame and return visualization and detections"""
        start_time = time.time()
        
        # Create a copy for visualization
        visualization = frame.copy()
        
        # Process with nomeroff-net
        results = self.pipeline([frame])
        (images, bboxs, points, zones, 
         region_ids, region_names, 
         count_lines, confidences, texts) = unzip(results)
        
        detections = []
        
        # Draw the detections
        if bboxs and len(bboxs[0]) > 0:
            # Create a list to pair bboxes with texts
            detection_pairs = list(zip(bboxs[0], texts[0]))
            
            for bbox, text in detection_pairs:
                # Get coordinates
                x1, y1, x2, y2 = map(int, bbox[:4])
                det_confidence = float(bbox[4])
                
                # Skip if confidence is too low
                if det_confidence < self.min_confidence:
                    continue
                
                # Ensure text is a string
                if isinstance(text, list):
                    text = ' '.join(text)
                
                # Draw bounding box and text
                cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                label = f"{text} ({det_confidence:.2f})"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                
                cv2.rectangle(visualization, 
                            (x1, y1 - text_size[1] - 10),
                            (x1 + text_size[0], y1),
                            (0, 255, 0),
                            -1)
                
                cv2.putText(visualization, 
                           label,
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1.0,
                           (0, 0, 0),
                           2)
                
                detections.append({
                    'text': text,
                    'confidence': det_confidence,
                    'bbox': bbox
                })
                
                print(f"Detected plate: {text} (confidence: {det_confidence:.2f})")
        
        # Calculate and store processing time
        process_time = time.time() - start_time
        self.processing_times.append(process_time)
        if len(self.processing_times) > self.max_processing_times:
            self.processing_times.pop(0)
        
        return visualization, detections

# [Continued in Part 2...]
# [Continuing from Part 1...]

    def process_image(self):
        """Process a single image"""
        print(f"\nProcessing image: {self.source_path}")
        
        # Read image
        frame = cv2.imread(self.source_path)
        if frame is None:
            raise ValueError(f"Could not read image: {self.source_path}")
        
        # Process frame
        visualization, detections = self.process_frame(frame)
        
        # Save visualization
        output_path = str(Path('output') / f"detected_{Path(self.source_path).name}")
        cv2.imwrite(output_path, visualization)
        print(f"Saved detection visualization to: {output_path}")
        
        # Store detections in database
        for detection in detections:
            self.db.store_detection(
                detection['text'],
                detection['confidence'],
                detection['bbox'],
                self.source_path,
                'image'
            )
        
        # Display result
        self.display_frame(visualization)
        print("\nPress 'q' to exit or 's' to save the current frame")
        while cv2.waitKey(0) not in [ord('q'), 27]:  # Wait for 'q' or ESC
            pass

    def process_video(self):
        """Process video file or camera stream"""
        # Open video capture
        if self.source_type == 'camera':
            print(f"\nOpening camera {self.camera_id}")
            cap = cv2.VideoCapture(self.camera_id)
        else:
            print(f"\nOpening video file: {self.source_path}")
            cap = cv2.VideoCapture(self.source_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video source")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.source_type == 'video':
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
        else:
            print(f"Camera properties: {frame_width}x{frame_height} @ {fps}fps")
        
        # Initialize video writer for saving output
        output_path = None
        writer = None
        if self.source_type == 'video':
            output_path = str(Path('output') / f"detected_{Path(self.source_path).name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        try:
            print("\nProcessing... Press 'q' to quit, 's' to save current frame")
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    if self.source_type == 'video':
                        break
                    continue
                
                # Skip frames according to frame_skip setting
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                visualization, detections = self.process_frame(frame)
                
                # Calculate FPS
                if self.processing_times:
                    avg_time = np.mean(self.processing_times[-30:])  # Average over last 30 frames
                    current_fps = 1.0 / avg_time if avg_time > 0 else 0
                    
                    # Add FPS info to frame
                    cv2.putText(visualization, 
                               f"FPS: {current_fps:.1f}",
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1.0,
                               (0, 255, 0),
                               2)
                
                # Store detections in database
                for detection in detections:
                    self.db.store_detection(
                        detection['text'],
                        detection['confidence'],
                        detection['bbox'],
                        self.source_path or f"camera_{self.camera_id}",
                        self.source_type
                    )
                
                # Save frame if we're processing a video file
                if writer is not None:
                    writer.write(visualization)
                
                # Display frame
                if not self.display_frame(visualization):
                    break
                
                # Update progress for video files
                frame_count += 1
                if self.source_type == 'video':
                    progress = (frame_count / total_frames) * 100
                    print(f"\rProgress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end="")
        
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    def display_frame(self, frame):
        """Display a frame and handle keyboard input"""
        window_name = 'License Plate Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Calculate window size
        height, width = frame.shape[:2]
        window_width = int(width * self.display_scale)
        window_height = int(height * self.display_scale)
        
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('s'):
            save_path = f"output/saved_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(save_path, frame)
            print(f"\nSaved frame to: {save_path}")
        
        return True

    def run(self):
        """Main processing loop"""
        try:
            if self.source_type == 'image':
                self.process_image()
            else:
                self.process_video()
            
            # Display statistics
            stats = self.db.get_statistics()
            print("\nProcessing Statistics:")
            print(f"Total detections: {stats['total_detections']}")
            print(f"Average confidence: {stats['average_confidence']:.2f}")
            print("\nDetections by source type:")
            for source, count in stats['detections_by_source'].items():
                print(f"  {source}: {count}")
            
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='License Plate Detection System')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'camera'],
                      required=True, help='Processing mode')
    parser.add_argument('--source', type=str,
                      help='Path to image or video file (not needed for camera mode)')
    parser.add_argument('--camera-id', type=int, default=0,
                      help='Camera device ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5,
                      help='Minimum confidence threshold (default: 0.5)')
    parser.add_argument('--frame-skip', type=int, default=100,
                      help='Process every nth frame (default: 2)')
    parser.add_argument('--display-scale', type=float, default=0.7,
                      help='Display window scale (default: 0.7)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['image', 'video'] and not args.source:
        parser.error(f"{args.mode} mode requires --source argument")
    
    # Initialize detector
    detector = LicensePlateDetector(
        source_type=args.mode,
        source_path=args.source,
        camera_id=args.camera_id
    )
    
    # Update settings from command line arguments
    detector.min_confidence = args.confidence
    detector.frame_skip = args.frame_skip
    detector.display_scale = args.display_scale
    
    # Run detection
    detector.run()

if __name__ == "__main__":
    main()