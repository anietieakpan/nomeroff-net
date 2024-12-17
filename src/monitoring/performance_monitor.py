# src/monitoring/performance_monitor.py
from typing import Dict, Any
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors and tracks performance metrics"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.processing_times = []
        self.detection_counts = []
        self.start_time = time.time()
        self.frames_processed = 0
        logger.info("Initialized PerformanceMonitor")

    def update_metrics(self, num_detections: int) -> None:
        """Update performance metrics"""
        self.frames_processed += 1
        self.detection_counts.append(num_detections)
        
        if len(self.detection_counts) > self.window_size:
            self.detection_counts.pop(0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        elapsed_time = time.time() - self.start_time
        
        stats = {
            'elapsed_time': elapsed_time,
            'frames_processed': self.frames_processed,
            'avg_detections': np.mean(self.detection_counts) if self.detection_counts else 0,
            'fps': self.frames_processed / elapsed_time if elapsed_time > 0 else 0
        }
        
        logger.debug(f"Performance stats: {stats}")
        return stats
