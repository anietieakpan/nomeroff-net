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
                if det_confidence < self.config['min_confidence']:
                    continue
                
                # Ensure text is a string
                if isinstance(text, list):
                    text = ' '.join(text)
                
                # Draw detection
                if self.config['show_display']:
                    cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Add background for text
                    label = f"{text} ({det_confidence:.2f})"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    cv2.rectangle(visualization, 
                                (x1, y1 - text_size[1] - 10),
                                (x1 + text_size[0], y1),
                                (0, 255, 0),
                                -1)
                    
                    # Draw text
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
                    'bbox': [x1, y1, x2, y2]
                })
                
                print(f"Detected plate: {text} (confidence: {det_confidence:.2f})")

        # Update performance metrics
        process_time = time.time() - start_time
        self.performance.update(process_time, len(detections))
        
        # Add performance info if debug mode is enabled
        if self.config['show_display'] and self.config['debug_mode']:
            stats = self.performance.get_stats()
            debug_info = [
                f"FPS: {stats['fps']:.1f}",
                f"Process Time: {stats['avg_process_time']*1000:.1f}ms",
                f"Total Detections: {stats['total_detections']}",
                f"Detection Rate: {stats['detection_rate']*100:.1f}%"
            ]
            
            y_offset = 30
            for text in debug_info:
                cv2.putText(visualization, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 25
        
        return visualization, detections

# [Continued in next part...]

    # [Continuing LicensePlateDetector class...]

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
            print("\nProcessing... Press 'q' to quit, 's' to save current frame, 'd' to toggle debug info")
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    if self.source_type == 'video':
                        break
                    continue
                
                # Process frame if it meets skip criterion
                if frame_count % self.config['frame_skip'] == 0:
                    # Process frame
                    visualization, detections = self.process_frame(frame)
                    
                    # Store detections
                    if self.config['save_detections']:
                        for detection in detections:
                            self.db.store_detection(
                                detection['text'],
                                detection['confidence'],
                                detection['bbox'],
                                self.source_path or f"camera_{self.camera_id}",
                                self.source_type
                            )
                    
                    # Save processed frame if enabled
                    if writer is not None:
                        writer.write(visualization)
                    
                    # Display frame if enabled
                    if self.config['show_display']:
                        if not self.display_frame(visualization):
                            break
                else:
                    # Display original frame if not processing
                    if self.config['show_display']:
                        if not self.display_frame(frame):
                            break
                
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
            return True
        
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
            return False
        elif key == ord('s'):
            save_path = f"output/saved_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(save_path, frame)
            print(f"\nSaved frame to: {save_path}")
        elif key == ord('d'):
            self.config['debug_mode'] = not self.config['debug_mode']
            print(f"\nDebug mode: {'enabled' if self.config['debug_mode'] else 'disabled'}")
        
        return True

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
            print("\nDatabase Statistics:")
            print("-" * 50)
            print(f"Total stored detections: {db_stats['total_detections']}")
            print(f"Average confidence: {db_stats['average_confidence']:.2f}")
            print("\nDetections by source:")
            for source, count in db_stats['detections_by_source'].items():
                print(f"  {source}: {count}")

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
            
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            if self.config['debug_mode']:
                import traceback
                traceback.print_exc()
        finally:
            cv2.destroyAllWindows()

# [Continued in next part...]


def main():
    parser = argparse.ArgumentParser(description='License Plate Detection System')
    
    # Required arguments
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'camera'],
                      required=True, help='Processing mode')
    parser.add_argument('--source', type=str,
                      help='Path to image or video file (not needed for camera mode)')
    
    # Optional arguments
    parser.add_argument('--camera-id', type=int, default=0,
                      help='Camera device ID (default: 0)')
    parser.add_argument('--frame-skip', type=int, default=4,
                      help='Process every nth frame (default: 4)')
    parser.add_argument('--display-scale', type=float, default=0.6,
                      help='Display window scale (default: 0.6)')
    parser.add_argument('--process-scale', type=float, default=1.0,
                      help='Processing scale (default: 1.0)')
    parser.add_argument('--confidence', type=float, default=0.5,
                      help='Minimum confidence threshold (default: 0.5)')
    
    # Flags
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
    
    # Prepare configuration
    config = {
        'frame_skip': args.frame_skip,
        'display_scale': args.display_scale,
        'process_scale': args.process_scale,
        'min_confidence': args.confidence,
        'show_display': not args.no_display,
        'save_video': not args.no_save,
        'debug_mode': args.debug,
        'quiet_mode': args.quiet
    }
    
    # Print configuration if not in quiet mode
    if not args.quiet:
        print("\nLicense Plate Detection System")
        print("-" * 50)
        print(f"Mode: {args.mode}")
        print(f"Source: {args.source if args.source else f'Camera {args.camera_id}'}")
        print("\nConfiguration:")
        print(f"Frame Skip: {config['frame_skip']}")
        print(f"Display Scale: {config['display_scale']}")
        print(f"Process Scale: {config['process_scale']}")
        print(f"Confidence Threshold: {config['min_confidence']}")
        print(f"Show Display: {config['show_display']}")
        print(f"Save Video: {config['save_video']}")
        print(f"Debug Mode: {config['debug_mode']}")
        print("-" * 50)
    
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



