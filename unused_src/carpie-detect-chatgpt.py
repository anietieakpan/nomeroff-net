import cv2
from nomeroff_net.pipelines import DetectionPipeline
from nomeroff_net.tools import unzip
import sqlite3
import pandas as pd

# Initialize SQLite database
def initialize_db():
    conn = sqlite3.connect("license_plates.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# Save license plates to the database
def save_to_db(plates):
    conn = sqlite3.connect("license_plates.db")
    cursor = conn.cursor()
    for plate, timestamp in plates:
        cursor.execute("INSERT INTO plates (plate, timestamp) VALUES (?, ?)", (plate, timestamp))
    conn.commit()
    conn.close()

# Save license plates to an Excel file
def save_to_excel(plates, filename="license_plates.xlsx"):
    df = pd.DataFrame(plates, columns=["Plate", "Timestamp"])
    df.to_excel(filename, index=False)

# Process video for license plate detection
def process_video(video_path, output_excel=True):
    # Initialize Nomeroff Net
    pipeline = DetectionPipeline()
    pipeline.set_image_processing_params(skip_ocr=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    plates_detected = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect license plates
        detections = pipeline(frame)
        for detection in detections:
            plate_text = detection["text"]
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)  # Timestamp in milliseconds
            plates_detected.append((plate_text, f"{timestamp/1000:.2f}s"))

            # Display detected plate on the frame
            bbox = detection["bbox"]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame (optional)
        cv2.imshow("License Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Save results
    save_to_db(plates_detected)
    if output_excel:
        save_to_excel(plates_detected)

if __name__ == "__main__":
    initialize_db()
    video_file = "path/to/your/video.mp4"  # Replace with your video path
    process_video(video_file)
