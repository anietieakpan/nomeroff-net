import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger('license_plate_detector')

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters"""
    box_thickness: int = 2
    text_scale: float = 0.8
    text_thickness: int = 2
    overlay_alpha: float = 0.7
    debug_mode: bool = False
    show_persistence: bool = True
    color_scheme: str = 'dynamic'  # 'dynamic' or 'static'

class DetectionVisualizer:
    """Handle visualization of license plate detections"""
    
    def __init__(self, config: Dict):
        """
        Initialize visualizer
        
        Args:
            config: Visualization configuration dictionary
        """
        self.config = VisualizationConfig(
            box_thickness=config.get('box_thickness', 2),
            text_scale=config.get('text_scale', 0.8),
            text_thickness=config.get('text_thickness', 2),
            overlay_alpha=config.get('overlay_alpha', 0.7),
            debug_mode=config.get('debug_mode', False),
            show_persistence=config.get('show_persistence', True),
            color_scheme=config.get('color_scheme', 'dynamic')
        )
        
        # Color palette for static color scheme
        self.color_palette = [
            (0, 255, 0),    # Green
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (255, 0, 0),    # Red
            (0, 255, 255),  # Yellow
            (128, 0, 255)   # Purple
        ]
        
        logger.debug("Initialized detection visualizer")

    def draw_detections(self, 
                       frame: np.ndarray,
                       detections: List[Dict],
                       performance_stats: Optional[Dict] = None) -> np.ndarray:
        """
        Draw detections on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            performance_stats: Optional performance statistics to display
            
        Returns:
            Frame with visualizations
        """
        try:
            visualization = frame.copy()
            
            # Draw each detection
            for idx, detection in enumerate(detections):
                self._draw_single_detection(visualization, detection, idx)
            
            # Add debug overlay if enabled
            if self.config.debug_mode and performance_stats:
                self._add_debug_overlay(visualization, performance_stats)
            
            return visualization
            
        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return frame

    def _draw_single_detection(self, 
                             frame: np.ndarray,
                             detection: Dict,
                             index: int) -> None:
        """Draw single detection with box and label"""
        try:
            # Get detection info
            bbox = detection['bbox']
            text = detection['text']
            confidence = detection['confidence']
            persistence = detection.get('persistence_count', 1)
            is_new = detection.get('is_new', False)
            
            # Get color for detection
            color = self._get_detection_color(
                persistence, 
                is_new, 
                index
            )
            
            # Draw bounding box
            self._draw_box(frame, bbox, color)
            
            # Draw label
            self._draw_label(frame, bbox, text, confidence, 
                           persistence, color)
                           
        except Exception as e:
            logger.error(f"Error drawing detection: {str(e)}")

    def _get_detection_color(self, 
                           persistence: int,
                           is_new: bool,
                           index: int) -> Tuple[int, int, int]:
        """Get color for detection based on configuration"""
        if self.config.color_scheme == 'static':
            return self.color_palette[index % len(self.color_palette)]
            
        # Dynamic color scheme
        if is_new:
            return (0, 255, 0)  # Bright green for new detections
            
        # Fade from green to blue based on persistence
        alpha = min(persistence / 15, 1.0)  # Adjust 15 based on max persistence
        return (
            int(255 * (1 - alpha)),  # R
            int(255 * (1 - alpha)),  # G
            int(255 * alpha)         # B
        )

    def _draw_box(self, 
                  frame: np.ndarray,
                  bbox: List[int],
                  color: Tuple[int, int, int]) -> None:
        """Draw bounding box with semi-transparent effect"""
        x1, y1, x2, y2 = bbox
        
        # Create overlay for semi-transparent effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 
                     self.config.box_thickness)
        
        # Blend overlay with original frame
        cv2.addWeighted(
            overlay, self.config.overlay_alpha,
            frame, 1 - self.config.overlay_alpha,
            0, frame
        )

    def _draw_label(self,
                    frame: np.ndarray,
                    bbox: List[int],
                    text: str,
                    confidence: float,
                    persistence: int,
                    color: Tuple[int, int, int]) -> None:
        """Draw text label with background"""
        x1, y1 = bbox[0], bbox[1]
        
        # Prepare label text
        label = f"{text} ({confidence:.2f})"
        if self.config.show_persistence:
            label += f" [{persistence}]"
        
        # Calculate text size
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale,
            self.config.text_thickness
        )[0]
        
        # Draw text background
        cv2.rectangle(
            frame,
            (x1, y1 - text_size[1] - 10),
            (x1 + text_size[0], y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale,
            (255, 255, 255),  # White text
            self.config.text_thickness
        )

    def _add_debug_overlay(self, 
                          frame: np.ndarray,
                          stats: Dict) -> None:
        """Add debug information overlay"""
        debug_info = [
            f"FPS: {stats.get('fps', 0):.1f}",
            f"Processing Time: {stats.get('avg_process_time', 0)*1000:.1f}ms",
            f"Active Detections: {stats.get('active_detections', 0)}",
            f"Total Detections: {stats.get('total_detections', 0)}",
            f"Unique Plates: {stats.get('unique_plates', 0)}"
        ]
        
        overlay = frame.copy()
        padding = 10
        line_height = 25
        start_x = padding
        start_y = padding
        
        # Draw background
        max_width = max([cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, 2)[0][0] for text in debug_info])
        cv2.rectangle(
            overlay,
            (start_x, start_y),
            (start_x + max_width + padding * 2,
             start_y + len(debug_info) * line_height + padding),
            (0, 0, 0),
            -1
        )
        
        # Blend overlay
        cv2.addWeighted(
            overlay, 0.7,
            frame, 0.3,
            0, frame
        )
        
        # Draw text
        for i, text in enumerate(debug_info):
            y = start_y + (i + 1) * line_height
            cv2.putText(
                frame,
                text,
                (start_x + padding, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )