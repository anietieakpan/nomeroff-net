from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger('license_plate_detector')

@dataclass
class Detection:
    """Data class for storing detection information"""
    text: str
    bbox: List[int]
    confidence: float
    frames_since_update: int = 0
    persistence_count: int = 1
    first_seen: datetime = None
    is_new: bool = True
    updated_this_frame: bool = True

    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = datetime.now()

class DetectionTracker:
    """Track and manage license plate detections across frames"""
    
    def __init__(self, max_persistence: int = 15, iou_threshold: float = 0.5):
        """
        Initialize the tracker
        
        Args:
            max_persistence: Maximum number of frames to track a detection
            iou_threshold: IoU threshold for matching detections
        """
        self.max_persistence = max_persistence
        self.iou_threshold = iou_threshold
        self.active_detections: Dict[str, Detection] = {}
        logger.debug("Initialized detection tracker")

    def update(self, new_detections: List[Dict]) -> List[Detection]:
        """
        Update tracker with new detections
        
        Args:
            new_detections: List of new detections to process
            
        Returns:
            List of active detections
        """
        # Update frame counts for existing detections
        for detection in self.active_detections.values():
            detection.frames_since_update += 1
            detection.updated_this_frame = False
            detection.is_new = False

        # Process new detections
        for new_det in new_detections:
            plate_text = new_det['text']
            matched = False

            # Check if this is an update to existing detection
            if plate_text in self.active_detections:
                existing_det = self.active_detections[plate_text]
                if self._check_detection_match(new_det, existing_det):
                    self._update_existing_detection(existing_det, new_det)
                    matched = True

            # If no match found, create new detection
            if not matched:
                self._add_new_detection(new_det)

        # Remove old detections
        self._cleanup_detections()

        return self.get_active_detections()

    def _check_detection_match(self, new_det: Dict, existing_det: Detection) -> bool:
        """Check if new detection matches existing one"""
        iou = self.calculate_iou(new_det['bbox'], existing_det.bbox)
        return iou > self.iou_threshold

    def _update_existing_detection(self, existing_det: Detection, new_det: Dict):
        """Update existing detection with new information"""
        existing_det.bbox = new_det['bbox']
        existing_det.confidence = max(existing_det.confidence, new_det['confidence'])
        existing_det.frames_since_update = 0
        existing_det.persistence_count += 1
        existing_det.updated_this_frame = True
        logger.debug(f"Updated existing detection: {existing_det.text}")

    def _add_new_detection(self, new_det: Dict):
        """Add new detection to tracker"""
        detection = Detection(
            text=new_det['text'],
            bbox=new_det['bbox'],
            confidence=new_det['confidence']
        )
        self.active_detections[new_det['text']] = detection
        logger.debug(f"Added new detection: {detection.text}")

    def _cleanup_detections(self):
        """Remove old detections"""
        to_remove = []
        for text, detection in self.active_detections.items():
            if detection.frames_since_update >= self.max_persistence:
                to_remove.append(text)
        
        for text in to_remove:
            del self.active_detections[text]
            logger.debug(f"Removed stale detection: {text}")

    @staticmethod
    def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1: First bounding box coordinates [x1, y1, x2, y2]
            bbox2: Second bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            IoU score between 0 and 1
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        # Check vertical displacement
        center_y1 = (bbox1[1] + bbox1[3]) / 2
        center_y2 = (bbox2[1] + bbox2[3]) / 2
        if abs(center_y1 - center_y2) > min(bbox1[3] - bbox1[1], bbox2[3] - bbox2[1]):
            return 0

        return intersection / union if union > 0 else 0

    def get_active_detections(self) -> List[Detection]:
        """Get list of currently active detections"""
        return [det for det in self.active_detections.values() 
                if det.updated_this_frame]