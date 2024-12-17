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
    # [Previous DB class implementation remains the same]
    # ... [DB class code]

class LicensePlateDetector:
    def __init__(self, source_type='image', source_path=None, camera_id=0):
        self.source_type = source_type
        self.source_path = source_path
        self.camera_id = camera_id
        
        # Initialize pipeline
        print("Loading Nomeroff-net pipeline...")
        self.pipeline = pipeline("number_plate_detection_and_reading")
        print("Pipeline loaded successfully")
        
        # Create output directory
        Path('output').mkdir(exist_ok=True)
        
        # Initialize database
        self.db = LicensePlateDB()

    def process_frame(self, frame):
        """Process a single frame and return visualization and detections"""
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
        
        return visualization, detections

    def process_image(self):
        """Process a single image"""
        print(f"Processing image: {self.source_path}")
        
        # Read image
        frame = cv2.imread(self.source_path)
        if frame is None:
            raise ValueError(f"Could not read image: {self.source_path}")
        
        # Process frame
        visualization, detections = self.process_frame(frame)
        
        # Save visualization
        output_path = str(Path('output') / f"detected_{Path(self.source_path).name}")
        cv2.imwrite(output_path, visualization)
        
        # Store detections in database
        for detection in detections:
            self.db.store_detection(
                detection['text'],
                detection['confidence'],
                detection['bbox'],
                self.source_path
            )
        
        # Display result
        self.display_frame(visualization)

    def process_video(self):
        """Process video file or camera stream"""
        # Open video capture
        if self.source_type == 'camera':
            print(f"Opening camera {self.camera_id}")
            cap = cv2.VideoCapture(self.camera_id)
        else:
            print(f"Opening video file: {self.source_path}")
            cap = cv2.VideoCapture(self.source_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video source")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if self.source_type == 'video':
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video FPS: {fps}, Total frames: {total_frames}")
        
        # Initialize video writer for saving output
        output_path = None
        writer = None
        if self.source_type == 'video':
            output_path = str(Path('output') / f"detected_{Path(self.source_path).name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        processing_times = []
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    if self.source_type == 'video':
                        break
                    continue
                
                start_time = time.time()
                
                # Process frame
                visualization, detections = self.process_frame(frame)
                
                # Calculate processing time
                process_time = time.time() - start_time
                processing_times.append(process_time)
                avg_time = np.mean(processing_times[-30:])  # Average over last 30 frames
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
                        self.source_path or f"camera_{self.camera_id}"
                    )
                
                # Save frame if we're processing a video file
                if writer is not None:
                    writer.write(visualization)
                
                # Display frame
                self.display_frame(visualization)
                
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
        
        # Calculate window size (70% of original image size)
        height, width = frame.shape[:2]
        window_width = int(width * 0.7)
        window_height = int(height * 0.7)
        
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
        except Exception as e:
            print(f"Error during processing: {str(e)}")
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
    
    # Run detection
    detector.run()

if __name__ == "__main__":
    main()