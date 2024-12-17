# src/detector/frame_processor.py
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
import logging

logger = logging.getLogger(__name__)

class FrameProcessor:
    """Handles processing of individual frames for license plate detection"""
    
    def __init__(self, min_confidence: float = 0.5):
        logger.info("Initializing Nomeroff-net pipeline")
        self.pipeline = pipeline("number_plate_detection_and_reading")
        self.min_confidence = min_confidence
        logger.info(f"Initialized FrameProcessor with min_confidence={min_confidence}")

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process a single frame and return detections"""
        try:
            print("Processing frame...") # Debug print
            results = self.pipeline([frame])
            detections = self._process_results(results)
            # logger.debug(f"Processed frame with {len(detections)} detections")
            print(f"Found {len(detections)} detections") # Debug print
            if detections:
                print(f"Detection examples: {detections[:2]}") # Debug print
            return detections
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            raise

    def _process_results(self, results: Tuple) -> List[Dict[str, Any]]:
        """Process raw results from the pipeline"""
        (_, bboxs, _, _, _, _, _, _, texts) = unzip(results)
        
        detections = []
        if not bboxs or len(bboxs[0]) == 0:
            return detections

        for bbox, text in zip(bboxs[0], texts[0]):
            try:
                detection = self._create_detection(bbox, text)
                if detection is not None:
                    detections.append(detection)
            except Exception as e:
                logger.warning(f"Error processing detection: {str(e)}")
                continue

        return detections

    def _create_detection(self, bbox: np.ndarray, text: str) -> Dict[str, Any]:
        """Create a detection dictionary from raw bbox and text"""
        confidence = float(bbox[4])
        if confidence < self.min_confidence:
            return None

        x1, y1, x2, y2 = map(int, bbox[:4])
        
        if isinstance(text, list):
            text = ' '.join(text)

        return {
            'text': text,
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence
        }