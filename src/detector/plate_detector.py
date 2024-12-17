# src/detector/plate_detector.py
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

from .frame_processor import FrameProcessor
from .visualization import DetectionVisualizer
from ..tracking.detection_tracker import DetectionTracker
from ..storage.base import DetectionStorage
from ..monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class LicensePlateDetector:
    """Main detector class that orchestrates the license plate detection process"""
    
    def __init__(self,
                 config: Dict[str, Any],
                 frame_processor: FrameProcessor,
                 visualizer: DetectionVisualizer,
                 tracker: DetectionTracker,
                 storage: Optional[DetectionStorage] = None):
        self.config = config
        self.frame_processor = frame_processor
        self.visualizer = visualizer
        self.tracker = tracker
        self.storage = storage
        self.performance = PerformanceMonitor()
        
        # Create output directory
        Path('output').mkdir(exist_ok=True)
        logger.info("Initialized LicensePlateDetector")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process a single frame and return the visualization"""
        try:
            # Process frame
            detections = self.frame_processor.process_frame(frame)
            
            # Update tracker
            self.tracker.update(detections)
            active_detections = self.tracker.get_active_detections()
            
            # Store new detections
            if self.storage:
                self._store_new_detections(active_detections)
            
            # Create visualization
            visualization = self.visualizer.draw_detections(frame, active_detections)
            
            # Update performance metrics
            self.performance.update_metrics(len(active_detections))
            
            return visualization, True
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, False

    def _store_new_detections(self, detections: list) -> None:
        """Store new detections in the database"""
        try:
            for detection in detections:
                if detection.get('is_new', False):
                    self.storage.store_detection(
                        plate_text=detection['text'],
                        confidence=detection['confidence'],
                        bbox=detection['bbox'],
                        image_path=str(self.config.get('source_path', 'unknown')),
                        source_type=self.config.get('source_type', 'unknown'),
                        persistence_count=detection.get('persistence_count', 1)
                    )
        except Exception as e:
            logger.error(f"Error storing detections: {str(e)}")