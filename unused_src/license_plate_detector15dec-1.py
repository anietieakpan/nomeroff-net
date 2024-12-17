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
    """Handles tracking and persistence of detections"""
    def __init__(self, max_persistence=15):
        self.detections = {}  # Format: {plate_text: detection_info}
        self.max_persistence = max_persistence
    
    def update(self, new_detections):
        """Update tracker with new detections"""
        # Mark all existing detections as not updated
        for det in self.detections.values():
            det['updated_this_frame'] = False
            det['frames_since_update'] += 1
        
        # Process new detections
        for new_det in new_detections:
            plate_text = new_det['text']
            
            if plate_text in self.detections:
                # Update existing detection
                det = self.detections[plate_text]
                det['bbox'] = new_det['bbox']
                det['confidence'] = max(det['confidence'], new_det['confidence'])
                det['frames_since_update'] = 0
                det['persistence_count'] += 1
                det['updated_this_frame'] = True
                det['is_new'] = False
            else:
                # Add new detection
                self.detections[plate_text] = {
                    'bbox': new_det['bbox'],
                    'confidence': new_det['confidence'],
                    'frames_since_update': 0,
                    'persistence_count': 1,
                    'first_seen': datetime.now(),
                    'is_new': True,
                    'updated_this_frame': True
                }
        
        # Remove old detections
        self.detections = {
            text: det for text, det in self.detections.items()
            if det['frames_since_update'] < self.max_persistence
        }
    
    def get_active_detections(self):
        """Get list of currently active detections"""
        active_dets = []
        for text, det in self.detections.items():
            if det['updated_this_frame']:
                active_dets.append({
                    'text': text,
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'persistence_count': det['persistence_count'],
                    'is_new': det['is_new']
                })
        return active_dets

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
            new_detections = [d for d in detections if d.get('is_new', False)]
            self.detection_count += len(new_detections)
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

### END OF PART 1 ###
# Continue with Part 2


### START OF PART 2 ###
# Add after Part 1

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

    def get_statistics(self):
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            cursor = conn.execute('SELECT COUNT(DISTINCT plate_text) FROM detections')
            stats['unique_plates'] = cursor.fetchone()[0]
            
            cursor = conn.execute('SELECT COUNT(*) FROM detections')
            stats['total_detections'] = cursor.fetchone()[0]
            
            cursor = conn.execute('''
                SELECT AVG(confidence) as avg_confidence,
                       AVG(persistence_count) as avg_persistence
                FROM detections
            ''')
            result = cursor.fetchone()
            stats['average_confidence'] = result[0] or 0
            stats['average_persistence'] = result[1] or 0
            
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
            'save_detections': True,      # Save detections to database
            'save_video': True,           # Save processed video
            'debug_mode': False,          # Show debug information
            'detection_persistence': 15,   # Number of frames to persist detections
            'iou_threshold': 0.5,         # IoU threshold for matching detections
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
        self.tracker = DetectionTracker(max_persistence=self.config['detection_persistence'])
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
        
        # Process detections
        current_detections = []
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
                    'bbox': [x1, y1, x2, y2],
                    'confidence': det_confidence
                })

        # Update tracker with new detections
        self.tracker.update(current_detections)
        
        # Get active detections for visualization
        active_detections = self.tracker.get_active_detections()
        
        # Draw detections and store new ones
        for detection in active_detections:
            if detection['is_new'] and self.config['save_detections']:
                self.db.store_detection(
                    detection['text'],
                    detection['confidence'],
                    detection['bbox'],
                    self.source_path or f"camera_{self.camera_id}",
                    self.source_type,
                    detection['persistence_count']
                )
                print(f"New plate detected: {detection['text']} (confidence: {detection['confidence']:.2f})")
        
        # Draw active detections
        self.draw_detections(visualization, active_detections)
        
        # Update performance metrics
        process_time = time.time() - start_time
        self.performance.update(process_time, active_detections)
        
        # Add debug info if enabled
        if self.config['show_display'] and self.config['debug_mode']:
            self.add_debug_info(visualization)
        
        return visualization, active_detections

### END OF PART 2 ###
# Continue with Part 3

### START OF PART 3 ###
# Continue adding to the LicensePlateDetector class

    def draw_detections(self, frame, detections):
        """Draw detections with improved visualization and tracking"""
        viz_config = self.config['visualization']
        
        # Sort detections by persistence count to draw most persistent ones last
        sorted_detections = sorted(detections, 
                                 key=lambda x: x['persistence_count'])
        
        for detection in sorted_detections:
            x1, y1, x2, y2 = detection['bbox']
            text = detection['text']
            confidence = detection['confidence']
            persistence = detection['persistence_count']
            is_new = detection['is_new']
            
            # Calculate opacity and color based on persistence and newness
            alpha = min(persistence / self.config['detection_persistence'], 1.0)
            
            if is_new:
                # New detections are bright green
                color = (0, 255, 0)
            else:
                # Existing detections fade from green to blue
                color = (
                    int(255 * (1 - alpha)),  # R
                    int(255 * (1 - alpha)),  # G
                    int(255 * alpha)         # B
                )
            
            # Create overlay for semi-transparent effects
            overlay = frame.copy()
            
            # Draw bounding box
            cv2.rectangle(overlay, 
                         (x1, y1), (x2, y2),
                         color,
                         viz_config['box_thickness'])
            
            # Prepare label
            if self.config['debug_mode']:
                label = f"{text} ({confidence:.2f}) [{persistence}]"
            else:
                label = f"{text} ({confidence:.2f})"
            
            # Calculate text dimensions
            text_scale = viz_config['text_scale']
            text_thickness = viz_config['text_thickness']
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
            
            # Draw text background
            text_bg_pts = np.array([
                [x1, y1 - text_height - 10],
                [x1 + text_width + 10, y1 - text_height - 10],
                [x1 + text_width + 10, y1],
                [x1, y1]
            ], np.int32)
            
            cv2.fillPoly(overlay, [text_bg_pts], color)
            
            # Blend overlay with original frame
            cv2.addWeighted(
                overlay, viz_config['overlay_alpha'],
                frame, 1 - viz_config['overlay_alpha'],
                0, frame)
            
            # Draw text
            cv2.putText(frame,
                       label,
                       (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       text_scale,
                       (255, 255, 255),  # White text
                       text_thickness)

    def add_debug_info(self, frame):
        """Add debug information overlay"""
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
                       (0, 255, 0),  # Green text
                       2)
            y_offset += line_height

    def handle_keyboard_input(self, frame):
        """Handle keyboard input for display window"""
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

### END OF PART 3 ###
# Continue with Part 4

### START OF PART 4 ###
# Continue adding to the LicensePlateDetector class

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
                        visualization, detections = self.process_frame(frame)
                    else:
                        visualization = frame.copy()
                        self.draw_detections(visualization, 
                                          self.tracker.get_active_detections())
                        if self.config['debug_mode']:
                            self.add_debug_info(visualization)
                
                    # Save processed frame if enabled
                    if writer is not None:
                        writer.write(visualization)
                
                    # Update progress
                    frame_count += 1
                    if self.pbar is not None:
                        self.pbar.update(1)
                
                # Display frame if enabled
                if self.config['show_display']:
                    window_name = 'License Plate Detection'
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    
                    # Scale window if needed
                    if self.config['display_scale'] != 1.0:
                        window_width = int(width * self.config['display_scale'])
                        window_height = int(height * self.config['display_scale'])
                        cv2.resizeWindow(window_name, window_width, window_height)
                    
                    cv2.imshow(window_name, visualization)
                    
                    # Handle keyboard input
                    key = self.handle_keyboard_input(visualization)
                    if key == 'quit':
                        break
                    elif key == 'pause':
                        paused = not paused
                        print(f"\nPlayback {'paused' if paused else 'resumed'}")
        
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

    def display_statistics(self):
        """Display processing statistics"""
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
            print("\nDatabase Summary:")
            print("-" * 50)
            print(f"Total Plates Recorded: {db_stats['unique_plates']}")
            print(f"Average Confidence: {db_stats['average_confidence']:.2f}")
            print(f"Average Persistence: {db_stats['average_persistence']:.1f} frames")

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
    
    # Feature toggles
    parser.add_argument('--no-display', action='store_true',
                      help='Disable visualization')
    parser.add_argument('--no-save', action='store_true',
                      help='Disable saving video output')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['image', 'video'] and not args.source:
        parser.error(f"{args.mode} mode requires --source argument")
    
    # Prepare configuration
    config = {
        'frame_skip': args.frame_skip,
        'display_scale': args.display_scale,
        'min_confidence': args.confidence,
        'show_display': not args.no_display,
        'save_video': not args.no_save,
        'debug_mode': args.debug,
        'detection_persistence': args.persistence
    }
    
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