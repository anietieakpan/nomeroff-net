# src/detector/visualization.py
from typing import Dict, Any, List
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DetectionVisualizer:
    """Handles visualization of license plate detections"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.viz_config = config.get('visualization', {})
        logger.info("Initialized DetectionVisualizer")

    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw all detections on the frame"""
        try:
            frame_copy = frame.copy()
            sorted_detections = sorted(detections, 
                                    key=lambda x: x['persistence_count'])
            
            for detection in sorted_detections:
                self._draw_single_detection(frame_copy, detection)
            
            return frame_copy
        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return frame

    def _draw_single_detection(self, frame: np.ndarray, 
                             detection: Dict[str, Any]) -> None:
        """Draw a single detection on the frame"""
        try:
            x1, y1, x2, y2 = detection['bbox']
            text = detection['text']
            confidence = detection['confidence']
            persistence = detection.get('persistence_count', 1)
            is_new = detection.get('is_new', False)

            color = self._get_detection_color(persistence, is_new)
            self._draw_bounding_box(frame, (x1, y1, x2, y2), color)
            self._draw_label(frame, text, confidence, persistence, 
                           (x1, y1), color)
        except Exception as e:
            logger.warning(f"Error drawing single detection: {str(e)}")

    def _get_detection_color(self, persistence: int, is_new: bool) -> tuple:
        """Calculate detection color based on persistence and newness"""
        if is_new:
            return (0, 255, 0)  # New detections are bright green
    
        # Get detection_persistence from config, default to 15 if not found
        max_persistence = self.config.get('detector', {}).get('detection_persistence', 15)
        alpha = min(persistence / max_persistence, 1.0)
        return (
            int(255 * (1 - alpha)),  # R
            int(255 * (1 - alpha)),  # G
            int(255 * alpha)         # B
        )

    def _draw_bounding_box(self, frame: np.ndarray, 
                          bbox: tuple, color: tuple) -> None:
        """Draw bounding box on frame"""
        x1, y1, x2, y2 = bbox
        thickness = self.viz_config.get('box_thickness', 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    def _draw_label(self, frame: np.ndarray, text: str, 
                   confidence: float, persistence: int, 
                   position: tuple, color: tuple) -> None:
        """Draw label with detection information"""
        x1, y1 = position
        text_scale = self.viz_config.get('text_scale', 0.8)
        text_thickness = self.viz_config.get('text_thickness', 2)
        
        # Prepare label text
        if self.config.get('debug_mode', False):
            label = f"{text} ({confidence:.2f}) [{persistence}]"
        else:
            label = f"{text} ({confidence:.2f})"

        # Calculate text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)

        # Draw text background
        self._draw_text_background(frame, label, (x1, y1), 
                                 (text_width, text_height), color)

        # Draw text
        cv2.putText(frame, label,
                   (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   text_scale,
                   (255, 255, 255),  # White text
                   text_thickness)

    def _draw_text_background(self, frame: np.ndarray, text: str, 
                            position: tuple, dimensions: tuple, 
                            color: tuple) -> None:
        """Draw semi-transparent background for text"""
        x1, y1 = position
        text_width, text_height = dimensions
        
        text_bg_pts = np.array([
            [x1, y1 - text_height - 10],
            [x1 + text_width + 10, y1 - text_height - 10],
            [x1 + text_width + 10, y1],
            [x1, y1]
        ], np.int32)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [text_bg_pts], color)
        
        alpha = self.viz_config.get('overlay_alpha', 0.7)
        cv2.addWeighted(
            overlay, alpha,
            frame, 1 - alpha,
            0, frame)