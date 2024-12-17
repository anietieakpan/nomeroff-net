# tests/test_detector.py
import pytest
import numpy as np
from src.detector.frame_processor import FrameProcessor

@pytest.fixture
def frame_processor():
    return FrameProcessor(min_confidence=0.5)

@pytest.fixture
def sample_frame():
    # Create a blank frame
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_frame_processor_initialization(frame_processor):
    assert frame_processor.min_confidence == 0.5
    assert frame_processor.pipeline is not None

def test_process_empty_frame(frame_processor, sample_frame):
    detections = frame_processor.process_frame(sample_frame)
    assert isinstance(detections, list)
    assert len(detections) == 0