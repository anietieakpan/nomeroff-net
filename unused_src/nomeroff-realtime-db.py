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
                    image_path TEXT
                )
            ''')
            conn.commit()
    
    def store_detection(self, plate_text, confidence, bbox, image_path):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO detections 
                (timestamp, plate_text, confidence, x1, y1, x2, y2, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                plate_text,
                confidence,
                int(bbox[0]), int(bbox[1]),
                int(bbox[2]), int(bbox[3]),
                image_path
            ))
            conn.commit()

    def get_all_detections(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM detections ORDER BY timestamp DESC')
            return cursor.fetchall()

def process_and_visualize(image_path, db):
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    print("Loading pipeline...")
    number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", 
                                                image_loader="opencv")
    
    print(f"Processing image: {image_path}")
    
    # Load image for visualization
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Process with nomeroff-net
    results = number_plate_detection_and_reading([image_path])
    (images, bboxs, points, zones, 
     region_ids, region_names, 
     count_lines, confidences, texts) = unzip(results)
    
    # Create a copy for visualization
    visualization = image.copy()
    
    print("\nDetection Results:")
    
    # Draw the detections
    if bboxs and len(bboxs[0]) > 0:
        # Create a list to pair bboxes with texts
        detection_pairs = list(zip(bboxs[0], texts[0]))
        
        for bbox, text in detection_pairs:
            # Get coordinates
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            det_confidence = float(bbox[4])
            
            # Ensure text is a string
            if isinstance(text, list):
                text = ' '.join(text)
            
            # Draw bounding box
            cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Prepare label with confidence
            label = f"{text} ({det_confidence:.2f})"
            
            # Get text size
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            
            # Draw text background
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
            
            print(f"Detected plate: {text}")
            print(f"Position: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"Detection confidence: {det_confidence:.2f}")
            
            # Store in database
            db.store_detection(text, det_confidence, bbox, image_path)
    
    # Save the visualization
    output_path = str(Path('output') / f"detected_{Path(image_path).name}")
    cv2.imwrite(output_path, visualization)
    print(f"\nVisualization saved to: {output_path}")
    
    try:
        # Create window with specific size
        window_name = 'License Plate Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Calculate window size (70% of original image size)
        height, width = visualization.shape[:2]
        window_width = int(width * 0.7)
        window_height = int(height * 0.7)
        
        # Set window size
        cv2.resizeWindow(window_name, window_width, window_height)
        
        # Show image
        cv2.imshow(window_name, visualization)
        print("\nPress 'q' to exit, 's' to save current frame")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f"output/saved_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(save_path, visualization)
                print(f"Saved frame to: {save_path}")
        
    except Exception as e:
        print(f"Warning: Could not display window: {str(e)}")
        print("Continuing without live display...")
    
    finally:
        cv2.destroyAllWindows()
    
    return texts, bboxs, confidences

def display_database_summary(db):
    print("\nRecent Detections from Database:")
    print("-" * 50)
    detections = db.get_all_detections()
    for detection in detections:
        print(f"Time: {detection[1]}")
        print(f"Plate: {detection[2]}")
        print(f"Confidence: {detection[3]:.2f}")
        print("-" * 50)

def main():
    # Initialize database
    db = LicensePlateDB()
    
    # Image path
    image_path = "/home/aniix/alprs/nomeroff-net/images/42dand1s51t01.jpg"
    
    try:
        texts, bboxs, confidences = process_and_visualize(image_path, db)
        
        print("\nDetection Summary:")
        if texts and len(texts[0]) > 0:
            for i, text in enumerate(texts[0]):
                if isinstance(text, list):
                    text = ' '.join(text)
                print(f"Plate {i+1}: {text}")
        
        display_database_summary(db)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()