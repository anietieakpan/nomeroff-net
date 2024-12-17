# src/tracking/detection_tracker.py
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DetectionTracker:
    """Handles tracking and persistence of license plate detections"""
    
    def __init__(self, max_persistence: int = 15):
        self.detections: Dict[str, Dict[str, Any]] = {}
        self.max_persistence = max_persistence
        logger.info(f"Initialized DetectionTracker with max_persistence={max_persistence}")

    def update(self, new_detections: List[Dict[str, Any]]) -> None:
        """Update tracker with new detections"""
        # Mark all existing detections as not updated
        for det in self.detections.values():
            det['updated_this_frame'] = False
            det['frames_since_update'] += 1
        
        # Process new detections
        for new_det in new_detections:
            plate_text = new_det['text']
            
            if plate_text in self.detections:
                self._update_existing_detection(plate_text, new_det)
            else:
                self._add_new_detection(plate_text, new_det)
        
        # Remove old detections
        self._cleanup_old_detections()

    def _update_existing_detection(self, plate_text: str, new_det: Dict[str, Any]) -> None:
        """Update an existing detection with new information"""
        det = self.detections[plate_text]
        det['bbox'] = new_det['bbox']
        det['confidence'] = max(det['confidence'], new_det['confidence'])
        det['frames_since_update'] = 0
        det['persistence_count'] += 1
        det['updated_this_frame'] = True
        det['is_new'] = False
        logger.debug(f"Updated existing detection: {plate_text}")

    def _add_new_detection(self, plate_text: str, new_det: Dict[str, Any]) -> None:
        """Add a new detection to the tracker"""
        self.detections[plate_text] = {
            'bbox': new_det['bbox'],
            'confidence': new_det['confidence'],
            'frames_since_update': 0,
            'persistence_count': 1,
            'first_seen': datetime.now(),
            'is_new': True,
            'updated_this_frame': True
        }
        logger.debug(f"Added new detection: {plate_text}")

    def _cleanup_old_detections(self) -> None:
        """Remove detections that haven't been updated recently"""
        old_count = len(self.detections)
        self.detections = {
            text: det for text, det in self.detections.items()
            if det['frames_since_update'] < self.max_persistence
        }
        removed_count = old_count - len(self.detections)
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} old detections")

    def get_active_detections(self) -> List[Dict[str, Any]]:
        """Get list of currently active detections"""
        active_dets = []
        for text, det in self.detections.items():
            if det['updated_this_frame']:
                active_dets.append({
                    'text': text,
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'persistence_count': det['persistence_count'],
                    'is_new': det['is_new']
                })
        return active_dets