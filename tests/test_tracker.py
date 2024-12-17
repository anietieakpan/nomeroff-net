# tests/test_tracker.py
import pytest
from src.tracking.detection_tracker import DetectionTracker

@pytest.fixture
def tracker():
    return DetectionTracker(max_persistence=15)

@pytest.fixture
def sample_detections():
    return [
        {
            'text': 'ABC123',
            'bbox': [100, 100, 200, 150],
            'confidence': 0.95
        },
        {
            'text': 'XYZ789',
            'bbox': [300, 300, 400, 350],
            'confidence': 0.85
        }
    ]

def test_tracker_initialization(tracker):
    assert tracker.max_persistence == 15
    assert len(tracker.detections) == 0

def test_update_new_detections(tracker, sample_detections):
    tracker.update(sample_detections)
    active = tracker.get_active_detections()
    assert len(active) == 2
    assert all(det['is_new'] for det in active)

def test_persistence(tracker, sample_detections):
    # Add initial detections
    tracker.update(sample_detections)
    
    # Update with empty list multiple times
    for _ in range(10):
        tracker.update([])
    
    # Should still have detections
    assert len(tracker.get_active_detections()) == 0
    assert len(tracker.detections) == 2
    
    # Update past persistence limit
    for _ in range(6):
        tracker.update([])
    
    # Should have no detections
    assert len(tracker.detections) == 0