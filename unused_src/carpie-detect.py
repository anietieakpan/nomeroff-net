import cv2
import numpy as np
from nomeroff_net import pipeline
from datetime import datetime
from pathlib import Path
import time

class LicensePlateDetector:
    def __init__(self):
        print("Initializing License Plate Detector...")
        
        # Initialize the detector pipeline
        print("Loading Nomeroff-net pipeline...")
        self.detector = pipeline("number_plate_detection_and_reading")
        print("Pipeline loaded successfully")
        
        # Create output directories
        Path('output').mkdir(exist_ok=True)
        
        # Configuration
        self.frame_skip = 5  # Process more frames
        self.resize_width = 1280  # Increased resolution
        self.detection_threshold = 0.2  # Lower threshold
        
        # Check GUI availability
        self.has_gui = True
        try:
            test_window = "Test"
            cv2.namedWindow(test_window)
            cv2.destroyWindow(test_window)
        except:
            print("GUI support not available. Will save output video only.")
            self.has_gui = False

    def enhance_frame(self, frame):
        """Enhance frame for better plate detection"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

    def resize_frame(self, frame):
        """Resize frame while maintaining aspect ratio"""
        height, width = frame.shape[:2]
        if width > self.resize_width:
            ratio = self.resize_width / width
            new_height = int(height * ratio)
            return cv2.resize(frame, (self.resize_width, new_height))
        return frame

    def process_video(self, video_path, output_path=None):
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video source: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps}fps")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detections_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_skip == 0:
                    # Enhance frame
                    process_frame = self.enhance_frame(frame.copy())
                    
                    # Resize for processing
                    process_frame = self.resize_frame(process_frame)
                    
                    try:
                        # Detect license plates
                        results = self.detector([process_frame])
                        
                        # Process detections
                        found_plates = False
                        for result in results:
                            if isinstance(result, dict):
                                bbox = result.get('bbox', [])
                                text = result.get('text', '')
                                confidence = result.get('confidence', 0.0)
                                
                                if bbox and confidence >= self.detection_threshold:
                                    found_plates = True
                                    detections_count += 1
                                    
                                    # Adjust coordinates for original frame size
                                    x1, y1, x2, y2 = map(int, bbox)
                                    
                                    if process_frame.shape != frame.shape:
                                        height_ratio = frame.shape[0] / process_frame.shape[0]
                                        width_ratio = frame.shape[1] / process_frame.shape[1]
                                        x1, x2 = int(x1 * width_ratio), int(x2 * width_ratio)
                                        y1, y2 = int(y1 * height_ratio), int(y2 * height_ratio)
                                    
                                    # Draw detection with thicker lines
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    
                                    # Add background for text
                                    text_size = cv2.getTextSize(f"{text} ({confidence:.2f})", 
                                                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                                    cv2.rectangle(frame, 
                                                (x1, y1 - text_size[1] - 10),
                                                (x1 + text_size[0], y1),
                                                (0, 255, 0),
                                                -1)
                                    
                                    # Draw text
                                    cv2.putText(frame, 
                                              f"{text} ({confidence:.2f})",
                                              (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              1.0,
                                              (0, 0, 0),
                                              2)
                                    
                                    print(f"Found plate: {text} (confidence: {confidence:.2f})")
                        
                        if writer:
                            writer.write(frame)
                    
                    except Exception as e:
                        print(f"Error processing frame: {str(e)}")
                
                if self.has_gui:
                    try:
                        cv2.imshow('License Plate Detection', frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):  # Save current frame
                            cv2.imwrite(f'output/frame_{frame_count}.jpg', frame)
                            print(f"Saved frame_{frame_count}.jpg")
                    except:
                        print("Error displaying frame. Continuing without visualization.")
                        self.has_gui = False
                
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    print(f"\rProcessed {frame_count}/{total_frames} frames ({fps:.2f} fps) - Found {detections_count} plates", end="")
        
        finally:
            print("\nCleaning up...")
            cap.release()
            if writer:
                writer.release()
            if self.has_gui:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass

def main():
    detector = LicensePlateDetector()
    
    video_path = "/home/aniix/alprs/trafficvid/Car-Traffic.mp4"
    output_path = "output/processed_video.mp4"
    
    try:
        detector.process_video(
            video_path=video_path,
            output_path=output_path
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()