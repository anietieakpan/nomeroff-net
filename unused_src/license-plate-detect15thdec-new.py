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
            
            # Total detections
            cursor = conn.execute('SELECT COUNT(DISTINCT plate_text) FROM detections')
            stats['unique_plates'] = cursor.fetchone()[0]
            
            cursor = conn.execute('SELECT COUNT(*) FROM detections')
            stats['total_detections'] = cursor.fetchone()[0]
            
            # Detections by source type
            cursor = conn.execute('''
                SELECT source_type, COUNT(DISTINCT plate_text) 
                FROM detections 
                GROUP BY source_type
            ''')
            stats['detections_by_source'] = dict(cursor.fetchall())
            
            # Average confidence and persistence
            cursor = conn.execute('''
                SELECT AVG(confidence) as avg_confidence,
                       AVG(persistence_count) as avg_persistence
                FROM detections
            ''')
            result = cursor.fetchone()
            stats['average_confidence'] = result[0]
            stats['average_persistence'] = result[1]
            
            return stats

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
            'debug_mode': False,          # Show debug information
            'detection_persistence': 15,   # Number of frames to persist detections
            'iou_threshold': 0.5,         # IoU threshold for matching detections
            'text_similarity_threshold': 0.8  # Threshold for matching plate text
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
        
        # Initialize detection tracking
        self.recent_detections = []  # List of currently tracked detections
        self.detection_history = []  # History of all detections
        
        # Initialize progress bar
        self.pbar = None

    def process_frame(self, frame):
        """Process a single frame and return visualization and detections"""
        start_time = time.time()
        
        # Create visualization frame
        visualization = frame.copy()
        
        # Process with nomeroff-net
        results = self.pipeline([frame])
        (images, bboxs, points, zones, 
         region_ids, region_names, 
         count_lines, confidences, texts) = unzip(results)
        
        current_detections = []
        
        # Process new detections
        if bboxs and len(bboxs[0]) > 0:
            detection_pairs = list(zip(bboxs[0], texts[0]))
            
            for bbox, text in detection_pairs:
                x1, y1, x2, y2 = map(int, bbox[:4])
                det_confidence = float(bbox[4])
                
                if det_confidence < self.config['min_confidence']:
                    continue
                
                if isinstance(text, list):
                    text = ' '.join(text)
                
                current_detections.append({
                    'text': text,
                    'confidence': det_confidence,
                    'bbox': [x1, y1, x2, y2],
                    'frames_since_detection': 0,
                    'persistence_count': 1
                })

        # Update detection history
        self.update_detections(current_detections)
        
        # Draw all active detections
        self.draw_detections(visualization)
        
        # Update performance metrics
        process_time = time.time() - start_time
        self.performance.update(process_time, len(current_detections), len(self.recent_detections))
        
        # Add debug info if enabled
        if self.config['show_display'] and self.config['debug_mode']:
            self.add_debug_info(visualization)
        
        return visualization, current_detections

    def update_detections(self, current_detections):
        """Update detection history and maintain persistent detections"""
        # Update frames_since_detection for existing detections
        for det in self.recent_detections:
            det['frames_since_detection'] += 1
        
        # Process new detections
        for new_det in current_detections:
            matched = False
            best_match = None
            best_iou = 0
            
            # Check for matches with existing detections
            for existing_det in self.recent_detections:
                iou = self.calculate_iou(new_det['bbox'], existing_det['bbox'])
                if iou > best_iou and iou > self.config['iou_threshold']:
                    best_match = existing_det
                    best_iou = iou
                    matched = True
            
            if matched and best_match:
                # Update existing detection
                best_match.update(new_det)
                best_match['frames_since_detection'] = 0
                best_match['persistence_count'] += 1
                
                # Update database if enabled
                if self.config['save_detections']:
                    self.db.update_persistence(best_match['text'], best_match['persistence_count'])
            else:
                # Add new detection
                self.recent_detections.append(new_det)
                if self.config['save_detections']:
                    self.db.store_detection(
                        new_det['text'],
                        new_det['confidence'],
                        new_det['bbox'],
                        self.source_path or f"camera_{self.camera_id}",
                        self.source_type,
                        new_det['persistence_count']
                    )
                print(f"New plate detected: {new_det['text']} (confidence: {new_det['confidence']:.2f})")
        
        # Remove old detections
        self.recent_detections = [det for det in self.recent_detections 
                                if det['frames_since_detection'] < self.config['detection_persistence']]

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union for two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def draw_detections(self, frame):
        """Draw all active detections on the frame"""
        for detection in self.recent_detections:
            x1, y1, x2, y2 = detection['bbox']
            text = detection['text']
            confidence = detection['confidence']
            persistence = detection['persistence_count']
            
            # Calculate color based on persistence (green -> yellow -> red)
            max_persistence = self.config['detection_persistence']
            persistence_ratio = min(persistence / max_persistence, 1.0)
            color = (0, 
                    int(255 * (1 - persistence_ratio)),  # Decrease green
                    int(255 * persistence_ratio))        # Increase red
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Add background for text
            label = f"{text} ({confidence:.2f}) [{persistence}]"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(frame, 
                        (x1, y1 - text_size[1] - 10),
                        (x1 + text_size[0], y1),
                        color,
                        -1)
            
            # Draw text
            cv2.putText(frame, 
                       label,
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.0,
                       (0, 0, 0),
                       2)

# [Continue with Part 3?]

    def add_debug_info(self, frame):
        """Add debug information to the frame"""
        stats = self.performance.get_stats()
        debug_info = [
            f"FPS: {stats['fps']:.1f}",
            f"Active Detections: {len(self.recent_detections)}",
            f"Total Detections: {stats['total_detections']}",
            f"Processing Time: {stats['avg_process_time']*1000:.1f}ms",
            f"Detection Rate: {stats['detection_rate']*100:.1f}%"
        ]
        
        y_offset = 30
        for text in debug_info:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25



    def process_video(self):
        """Process video file or camera stream"""
        # Open video source
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps")
        if self.source_type == 'video':
            print(f"Total frames: {total_frames}")
        
        # Initialize video writer if saving enabled
        writer = None
        if self.config['save_video'] and self.source_type == 'video':
            output_path = str(Path('output') / f"detected_{Path(self.source_path).name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")
        
        # Initialize progress bar for video files
        if self.source_type == 'video':
            self.pbar = tqdm(total=total_frames, desc="Processing", unit="frames")
        
        frame_count = 0
        try:
            print("\nProcessing... Controls:")
            print("  'q': Quit")
            print("  's': Save current frame")
            print("  'd': Toggle debug info")
            print("  'p': Toggle pause")
            
            paused = False
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        if self.source_type == 'video':
                            break
                        continue
                
                    # Process frame if it meets skip criterion
                    if frame_count % self.config['frame_skip'] == 0:
                        # Process frame
                        visualization, detections = self.process_frame(frame)
                    else:
                        # For skipped frames, just draw existing detections
                        visualization = frame.copy()
                        self.draw_detections(visualization)
                        if self.config['debug_mode']:
                            self.add_debug_info(visualization)
                    
                    # Save processed frame if enabled
                    if writer is not None:
                        writer.write(visualization)
                
                # Display frame if enabled
                if self.config['show_display']:
                    # Handle keyboard input
                    key = self.display_frame(visualization)
                    
                    if key == 'quit':
                        break
                    elif key == 'pause':
                        paused = not paused
                        print(f"\nPlayback {'paused' if paused else 'resumed'}")
                        continue
                
                if not paused:
                    # Update progress
                    frame_count += 1
                    if self.pbar is not None:
                        self.pbar.update(1)
        
        finally:
            # Cleanup
            if self.pbar is not None:
                self.pbar.close()
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
            # Display final statistics
            self.display_statistics()

    def display_frame(self, frame):
        """Display a frame and handle keyboard input"""
        if not self.config['show_display']:
            return None
        
        window_name = 'License Plate Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Scale window if needed
        if self.config['display_scale'] != 1.0:
            height, width = frame.shape[:2]
            window_width = int(width * self.config['display_scale'])
            window_height = int(height * self.config['display_scale'])
            cv2.resizeWindow(window_name, window_width, window_height)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 'quit'
        elif key == ord('s'):
            save_path = f"output/saved_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(save_path, frame)
            print(f"\nSaved frame to: {save_path}")
        elif key == ord('d'):
            self.config['debug_mode'] = not self.config['debug_mode']
            print(f"\nDebug mode: {'enabled' if self.config['debug_mode'] else 'disabled'}")
        elif key == ord('p'):
            return 'pause'
        
        return None

    def display_statistics(self):
        """Display processing statistics"""
        stats = self.performance.get_stats()
        print("\nProcessing Statistics:")
        print("-" * 50)
        print(f"Total time: {stats['elapsed_time']:.1f} seconds")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Average FPS: {stats['fps']:.1f}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Detection rate: {stats['detection_rate']*100:.1f}%")
        print(f"Average processing time: {stats['avg_process_time']*1000:.1f}ms per frame")
        
        if self.config['save_detections']:
            db_stats = self.db.get_statistics()
            print("\nDetection Summary:")
            print("-" * 50)
            print(f"Unique plates detected: {db_stats['unique_plates']}")
            print(f"Total detection events: {db_stats['total_detections']}")
            print(f"Average confidence: {db_stats['average_confidence']:.2f}")
            print(f"Average persistence: {db_stats['average_persistence']:.1f} frames")
            
            print("\nRecent Detections:")
            print("-" * 50)
            recent = self.db.get_recent_detections(5)
            for det in recent:
                print(f"Plate: {det[0]}")
                print(f"  Max Confidence: {det[1]:.2f}")
                print(f"  Detection Count: {det[2]}")
                print(f"  Max Persistence: {det[3]} frames")

# [Continue with Part 4?]
    def run(self):
        """Main processing loop"""
        try:
            if self.source_type == 'image':
                print(f"\nProcessing image: {self.source_path}")
                frame = cv2.imread(self.source_path)
                if frame is None:
                    raise ValueError(f"Could not read image: {self.source_path}")
                
                visualization, detections = self.process_frame(frame)
                
                # Save visualization
                output_path = str(Path('output') / f"detected_{Path(self.source_path).name}")
                cv2.imwrite(output_path, visualization)
                print(f"Saved result to: {output_path}")
                
                # Display result
                if self.config['show_display']:
                    print("\nPress any key to close...")
                    cv2.imshow('License Plate Detection', visualization)
                    cv2.waitKey(0)
            else:
                self.process_video()
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            if self.config['debug_mode']:
                import traceback
                traceback.print_exc()
        finally:
            cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser(
        description='Enhanced License Plate Detection System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'camera'],
                      required=True, help='Processing mode')
    parser.add_argument('--source', type=str,
                      help='Path to image or video file (not needed for camera mode)')
    
    # Processing settings
    parser.add_argument('--camera-id', type=int, default=0,
                      help='Camera device ID')
    parser.add_argument('--frame-skip', type=int, default=4,
                      help='Process every nth frame')
    parser.add_argument('--confidence', type=float, default=0.5,
                      help='Minimum confidence threshold')
    parser.add_argument('--persistence', type=int, default=15,
                      help='Number of frames to persist detections')
    
    # Display settings
    parser.add_argument('--display-scale', type=float, default=0.6,
                      help='Display window scale')
    parser.add_argument('--process-scale', type=float, default=1.0,
                      help='Processing scale')
    
    # Feature toggles
    parser.add_argument('--no-display', action='store_true',
                      help='Disable visualization')
    parser.add_argument('--no-save', action='store_true',
                      help='Disable saving video output')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    parser.add_argument('--quiet', action='store_true',
                      help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['image', 'video'] and not args.source:
        parser.error(f"{args.mode} mode requires --source argument")
    
    if args.frame_skip < 1:
        parser.error("Frame skip must be >= 1")
    
    if not (0.1 <= args.display_scale <= 1.0):
        parser.error("Display scale must be between 0.1 and 1.0")
    
    if not (0.1 <= args.process_scale <= 1.0):
        parser.error("Process scale must be between 0.1 and 1.0")
    
    if not (0.0 <= args.confidence <= 1.0):
        parser.error("Confidence threshold must be between 0.0 and 1.0")
    
    if args.persistence < 1:
        parser.error("Persistence must be >= 1")
    
    # Prepare configuration
    config = {
        'frame_skip': args.frame_skip,
        'display_scale': args.display_scale,
        'process_scale': args.process_scale,
        'min_confidence': args.confidence,
        'show_display': not args.no_display,
        'save_video': not args.no_save,
        'debug_mode': args.debug,
        'detection_persistence': args.persistence
    }
    
    # Print configuration if not in quiet mode
    if not args.quiet:
        print("\nLicense Plate Detection System")
        print("=" * 50)
        print(f"Mode: {args.mode}")
        print(f"Source: {args.source if args.source else f'Camera {args.camera_id}'}")
        print("\nConfiguration:")
        print("-" * 50)
        print(f"Frame Skip: {config['frame_skip']}")
        print(f"Display Scale: {config['display_scale']}")
        print(f"Process Scale: {config['process_scale']}")
        print(f"Confidence Threshold: {config['min_confidence']}")
        print(f"Detection Persistence: {config['detection_persistence']} frames")
        print(f"Show Display: {config['show_display']}")
        print(f"Save Video: {config['save_video']}")
        print(f"Debug Mode: {config['debug_mode']}")
        print("=" * 50)
    
    try:
        # Initialize and run detector
        detector = LicensePlateDetector(
            source_type=args.mode,
            source_path=args.source,
            camera_id=args.camera_id,
            config=config
        )
        
        detector.run()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        if config['debug_mode']:
            import traceback
            traceback.print_exc()
    finally:
        print("\nProcessing complete")

if __name__ == "__main__":
    main()

