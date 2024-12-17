# tests/test_storage.py
import pytest
from src.storage.sqlite_storage import SQLiteStorage

def test_store_detection(sqlite_storage, sample_detection):
    sqlite_storage.store_detection(
        plate_text=sample_detection['text'],
        confidence=sample_detection['confidence'],
        bbox=sample_detection['bbox'],
        image_path=sample_detection['image_path'],
        source_type=sample_detection['source_type']
    )
    
    stats = sqlite_storage.get_statistics()
    assert stats['unique_plates'] == 1
    assert abs(stats['average_confidence'] - sample_detection['confidence']) < 0.001

def test_statistics(sqlite_storage):
    # Store multiple detections
    detections = [
        ('ABC123', 0.95, [100, 100, 200, 150], 'test1.jpg', 'image'),
        ('XYZ789', 0.85, [150, 150, 250, 200], 'test2.jpg', 'image'),
        ('ABC123', 0.90, [120, 120, 220, 170], 'test3.jpg', 'image')
    ]
    
    for det in detections:
        sqlite_storage.store_detection(*det)
    
    stats = sqlite_storage.get_statistics()
    assert stats['unique_plates'] == 2  # ABC123 appears twice
    assert 0.85 <= stats['average_confidence'] <= 0.95