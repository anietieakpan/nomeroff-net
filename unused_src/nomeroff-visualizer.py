import cv2
import numpy as np
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
from pathlib import Path

def process_and_visualize(image_path):
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
        for bbox in bboxs[0]:
            # Get coordinates (bbox contains [x1, y1, x2, y2, confidence, class])
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            det_confidence = float(bbox[4])
            
            # Draw bounding box
            cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Get corresponding text and confidence
            for text in texts[0]:
                # Prepare text with confidence
                if isinstance(text, list):
                    text = ' '.join(text)  # Join list elements if text is a list
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
    
    # Save the visualization
    output_path = str(Path('output') / f"detected_{Path(image_path).name}")
    cv2.imwrite(output_path, visualization)
    print(f"\nVisualization saved to: {output_path}")
    
    return texts, bboxs, confidences

def main():
    # Image path
    image_path = "/home/aniix/alprs/nomeroff-net/images/Cars420.png"
    
    try:
        texts, bboxs, confidences = process_and_visualize(image_path)
        
        print("\nSummary of detections:")
        if texts and len(texts[0]) > 0:
            for i, text in enumerate(texts[0]):
                if isinstance(text, list):
                    text = ' '.join(text)  # Join list elements if text is a list
                print(f"Plate {i+1}: {text}")
        else:
            print("No license plates detected")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()