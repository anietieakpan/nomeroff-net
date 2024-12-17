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
from collections import OrderedDict

class DetectionTracker:
    """Tracks unique detections and their persistence"""
    def __init__(self, max_size=50):
        self.detections = OrderedDict()
        self.max_size = max_size
        
    def add_detection(self, plate_text, bbox, confidence):
        """Add or update a detection"""
        if plate_text in self.detections:
            det = self.detections[plate_text]
            det['frames_since_update'] = 0
            det['persistence_count'] += 1
            det['confidence'] = max(det['confidence'], confidence)
            det['bbox'] = bbox
            det['total_detections'] += 1
        else:
            if len(self.detections) >= self.max_size:
                # Remove oldest detection if at capacity
                self.detections.popitem(last=False)
            
            self.detections[plate_text] = {
                'bbox': bbox,
                'confidence': confidence,
                'frames_since_update': 0,
                'persistence_count': 1,
                'first_seen': datetime.now(),
                'total_detections': 1,
                'is_new': True
            }
            
        return self.detections[plate_text]
    
    def update_frames(self):
        """Update frame counts and remove stale detections"""
        to_remove = []
        for plate_text, det in self.detections.items():
            det['frames_since_update'] += 1
            det['is_new'] = False
            if det['frames_since_update'] > 30:  # Configurable threshold
                to_remove.append(plate_text)
        
        for plate_text in to_remove:
            del self.detections[plate_text]
    
    def get_active_detections(self):
        """Get list of currently active detections"""
        return [{'text': text, **data} 
                for text, data in self.detections.items()
                if data['frames_since_update'] < 15]  # Active threshold

class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.processing_times = []
        self.detection_count = 0
        self.frames_processed = 0
        self.start_time = time.time()
        self.active_detections = 0
        self.unique_plates = set()
        
    def update(self, process_time, detections=None):
        self.processing_times.append(process_time)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
        
        if detections:
            self.detection_count += len(detections)
            self.active_detections = len(detections)
            for det in detections:
                self.unique_plates.add(det['text'])
                
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
            'unique_plates': len(self.unique_plates),
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
                    persistence_count INTEGER,
                    total_detections INTEGER
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

# [Rest of the LicensePlateDB class methods remain the same as before]

### END OF PART 1 ###
# Continue with Part 2, which starts with the LicensePlateDetector class

### START OF PART 2 ###
# Add after Part 1

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
            'save_detections': True,      # Save detections to database
            'save_video': True,           # Save processed video
            'debug_mode': False,          # Show debug information
            'detection_persistence': 15,   # Number of frames to persist detections
            'max_tracked_plates': 50,     # Maximum number of plates to track
            'visualization': {
                'box_thickness': 2,
                'text_scale': 0.8,
                'text_thickness': 2,
                'overlay_alpha': 0.7
            }
        }
        
        # Update configuration with provided values
        if config:
            self.config.update(config)

        # Initialize pipeline
        print("\nInitializing License Plate Detection System...")
        print("Loading Nomeroff-net pipeline...")
        self.pipeline = pipeline("number_plate_detection_and_reading")
        print("Pipeline loaded successfully")

        # Initialize tracking and monitoring
        self.tracker = DetectionTracker(max_size=self.config['max_tracked_plates'])
        self.performance = PerformanceMonitor()
        
        # Create output directory
        Path('output').mkdir(exist_ok=True)
        
        # Initialize database if saving detections
        if self.config['save_detections']:
            self.db = LicensePlateDB()
        
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
        
        # Update tracking before processing new detections
        self.tracker.update_frames()
        
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
                
                # Update tracker and get detection info
                detection = self.tracker.add_detection(text, [x1, y1, x2, y2], det_confidence)
                
                # Add to current detections list
                current_detections.append({
                    'text': text,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': det_confidence,
                    'persistence_count': detection['persistence_count'],
                    'is_new': detection['is_new']
                })
                
                # Report new detections
                if detection['is_new']:
                    print(f"New plate detected: {text} (confidence: {det_confidence:.2f})")
                    
                    if self.config['save_detections']:
                        self.db.store_detection(
                            text,
                            det_confidence,
                            [x1, y1, x2, y2],
                            self.source_path or f"camera_{self.camera_id}",
                            self.source_type,
                            detection['total_detections']
                        )

        # Draw active detections
        self.draw_detections(visualization)
        
        # Update performance metrics
        process_time = time.time() - start_time
        self.performance.update(process_time, current_detections)
        
        # Add debug info if enabled
        if self.config['show_display'] and self.config['debug_mode']:
            self.add_debug_info(visualization)
        
        return visualization, current_detections

    def draw_detections(self, frame):
        """Draw all active detections with improved visualization"""
        viz_config = self.config['visualization']
        active_detections = self.tracker.get_active_detections()
        
        # Sort by persistence count to draw most persistent ones last
        active_detections.sort(key=lambda x: x['persistence_count'])
        
        for detection in active_detections:
            x1, y1, x2, y2 = detection['bbox']
            text = detection['text']
            confidence = detection['confidence']
            persistence = detection['persistence_count']
            
            # Calculate color based on persistence and confidence
            persistence_ratio = min(persistence / self.config['detection_persistence'], 1.0)
            confidence_ratio = min(confidence, 1.0)
            
            # Color scheme: Green for new detections, transitioning to blue for persistent ones
            color = (
                int(255 * (1 - persistence_ratio)),  # R
                int(255 * confidence_ratio),         # G
                int(255 * persistence_ratio)         # B
            )
            
            # Draw semi-transparent box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 
                         viz_config['box_thickness'])
            
            # Create label with plate text and metadata
            label = f"{text} ({confidence:.2f})"
            if self.config['debug_mode']:
                label += f" [{persistence}]"
            
            # Calculate text size and position
            text_scale = viz_config['text_scale']
            text_thickness = viz_config['text_thickness']
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                      text_scale, text_thickness)[0]
            
            # Draw text background
            text_bg_pts = np.array([
                [x1, y1 - text_size[1] - 10],
                [x1 + text_size[0], y1 - text_size[1] - 10],
                [x1 + text_size[0], y1],
                [x1, y1]
            ], np.int32)
            
            cv2.fillPoly(overlay, [text_bg_pts], color)
            
            # Blend overlay with original frame
            cv2.addWeighted(overlay, viz_config['overlay_alpha'], 
                          frame, 1 - viz_config['overlay_alpha'], 0, frame)
            
            # Draw text
            cv2.putText(frame, label,
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       text_scale,
                       (255, 255, 255),  # White text
                       text_thickness)

### END OF PART 2 ###
# Continue with Part 3
### START OF PART 3 ###
# Continue adding to the LicensePlateDetector class

    def add_debug_info(self, frame):
        """Add debug information overlay to frame"""
        stats = self.performance.get_stats()
        debug_info = [
            f"FPS: {stats['fps']:.1f}",
            f"Active Detections: {stats['active_detections']}",
            f"Unique Plates: {stats['unique_plates']}",
            f"Processing Time: {stats['avg_process_time']*1000:.1f}ms",
            f"Total Frames: {stats['frames_processed']}"
        ]
        
        # Create semi-transparent overlay for debug info
        overlay = frame.copy()
        padding = 10
        line_height = 25
        max_width = max([cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] 
                        for text in debug_info])
        
        # Draw background rectangle
        cv2.rectangle(overlay, 
                     (padding, padding), 
                     (max_width + padding * 2, len(debug_info) * line_height + padding * 2),
                     (0, 0, 0),
                     -1)
        
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        y_offset = padding + line_height
        for text in debug_info:
            cv2.putText(frame, text,
                       (padding, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 255, 0),
                       2)
            y_offset += line_height

    def process_video(self):
        """Process video file or camera stream with improved handling"""
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
        
        # Initialize video writer
        writer = None
        if self.config['save_video'] and self.source_type == 'video':
            output_path = str(Path('output') / f"detected_{Path(self.source_path).name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")
        
        # Initialize progress bar
        if self.source_type == 'video':
            self.pbar = tqdm(total=total_frames, desc="Processing", unit="frames")
        
        frame_count = 0
        try:
            print("\nControls:")
            print("  'q': Quit")
            print("  's': Save current frame")
            print("  'd': Toggle debug info")
            print("  'p': Toggle pause")
            print("  '+': Increase detection persistence")
            print("  '-': Decrease detection persistence")
            
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
                    elif key == 'persistence_up':
                        self.config['detection_persistence'] += 5
                        print(f"\nDetection persistence increased to: {self.config['detection_persistence']}")
                    elif key == 'persistence_down':
                        self.config['detection_persistence'] = max(5, self.config['detection_persistence'] - 5)
                        print(f"\nDetection persistence decreased to: {self.config['detection_persistence']}")
                
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

### END OF PART 3 ###
# Continue with Part 4

### START OF PART 4 ###
# Continue adding to the LicensePlateDetector class

    def display_frame(self, frame):
        """Display a frame and handle keyboard input with improved controls"""
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
        elif key == ord('+') or key == ord('='):
            return 'persistence_up'
        elif key == ord('-') or key == ord('_'):
            return 'persistence_down'
        
        return None

    def display_statistics(self):
        """Display comprehensive processing statistics"""
        stats = self.performance.get_stats()
        print("\nProcessing Statistics:")
        print("=" * 50)
        print(f"Time Elapsed: {stats['elapsed_time']:.1f} seconds")
        print(f"Frames Processed: {stats['frames_processed']}")
        print(f"Average FPS: {stats['fps']:.1f}")
        print(f"Unique Plates Detected: {stats['unique_plates']}")
        print(f"Total Detection Events: {stats['total_detections']}")
        print(f"Average Processing Time: {stats['avg_process_time']*1000:.1f}ms per frame")
        
        if self.config['save_detections']:
            db_stats = self.db.get_statistics()
            print("\nDetection Summary:")
            print("-" * 50)
            print(f"Total Plates in Database: {db_stats['unique_plates']}")
            print(f"Total Detection Records: {db_stats['total_detections']}")
            print(f"Average Confidence: {db_stats['average_confidence']:.2f}")
            print(f"Average Persistence: {db_stats['average_persistence']:.1f} frames")
            
            print("\nMost Recent Detections:")
            print("-" * 50)
            recent = self.db.get_recent_detections(5)
            for det in recent:
                print(f"Plate: {det[0]}")
                print(f"  Max Confidence: {det[1]:.2f}")
                print(f"  Detection Count: {det[2]}")
                print(f"  Max Persistence: {det[3]} frames")
                print("-" * 25)

    def run(self):
        """Main processing loop with improved error handling"""
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
        for key, value in config.items():
            print(f"{key}: {value}")
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

### END OF PART 4 ###