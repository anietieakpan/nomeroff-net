import cv2
import numpy as np
from pathlib import Path

def test_basic_operations(image_path):
    print("Testing basic image operations...")
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    # Step 1: Load image
    print(f"\nStep 1: Loading image from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    print(f"Image loaded successfully. Shape: {image.shape}")
    
    # Step 2: Basic transformations
    print("\nStep 2: Testing basic transformations")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output/1_grayscale.png", gray)
    print("Grayscale conversion successful")
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("output/2_blurred.png", blurred)
    print("Gaussian blur successful")
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite("output/3_edges.png", edges)
    print("Edge detection successful")
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")
    
    # Draw contours
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("output/4_contours.png", contour_image)
    print("Contour detection successful")
    
    # Test drawing functions
    test_image = image.copy()
    # Draw a rectangle
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), 2)
    # Draw some text
    cv2.putText(test_image, "Test Text", (100, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("output/5_drawing_test.png", test_image)
    print("Drawing operations successful")
    
    print("\nAll basic operations completed successfully")
    print("\nCheck the output folder for the following files:")
    print("1. 1_grayscale.png - Grayscale conversion")
    print("2. 2_blurred.png - Gaussian blur")
    print("3. 3_edges.png - Edge detection")
    print("4. 4_contours.png - Contour detection")
    print("5. 5_drawing_test.png - Drawing test")

def main():
    image_path = "/home/aniix/alprs/nomeroff-net/images/Cars420.png"
    
    try:
        test_basic_operations(image_path)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()