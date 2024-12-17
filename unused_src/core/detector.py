import cv2
import numpy as np
from nomeroff_net import pipeline
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path

from .tracker import DetectionTracker
from utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger('license_plate_detector')

class LicensePlateDetector:
    """Main class for license plate detection and processing"""
    
    def __init__(self, config: Dict):
        """
        Initialize the detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._initialize_components()
        logger.info("Initialized License Plate Detector")

    def _initialize_components(self):
        """Initialize detector components"""
        try:
            # Initialize the pipeline
            logger.info("Loading Nomeroff-net pipeline...")
            self.pipeline = pipeline("number_plate_detection_and_reading")
            
            # Initialize tracker
            self.tracker = DetectionTracker(
                max_persistence=self.config['detector']['detection_persistence'],
                iou_threshold=0.5
            )
            
            # Initialize performance monitor
            self.performance = PerformanceMonitor()
            
            # Create output directories
            self._create_output_dirs()
            
        except Exception as e:
            logger.error(f"Failed to initialize detector: {str(e)}")
            raise

    def _create_output_dirs(self):
        """Create necessary output directories"""
        Path(self.config['output']['json_path']).mkdir(parents=True, exist_ok=True)
        Path(self.config['output']['video_path']).mkdir(parents=True, exist_ok=True)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (processed frame, list of detections)
        """
        try:
            start_time = time.time()
            
            # Create copy for processing
            process_frame = frame.copy()
            
            # Resize if needed
            if self.config['detector']['process_scale'] != 1.0:
                process_frame = self._resize_frame(process_frame)
            
            # Run detection pipeline
            results = self.pipeline([process_frame])
            detections = self._process_pipeline_results(results, frame.shape)
            
            # Update tracker
            active_detections = self.tracker.update(detections)
            
            # Update performance metrics
            process_time = time.time() - start_time
            self.performance.update(process_time, len(active_detections))
            
            return frame, active_detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, []

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for processing"""
        scale = self.config['detector']['process_scale']
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        return cv2.resize(frame, (width, height))

    def _process_pipeline_results(self, 
                                results: tuple, 
                                original_shape: tuple) -> List[Dict]:
        """Process results from Nomeroff-net pipeline"""
        (images, bboxs, points, zones, 
         region_ids, region_names, 
         count_lines, confidences, texts) = results
        
        detections = []
        if not bboxs or len(bboxs[0]) == 0:
            return detections

        for bbox, text in zip(bboxs[0], texts[0]):
            x1, y1, x2, y2 = map(int, bbox[:4])
            confidence = float(bbox[4])
            
            if confidence < self.config['detector']['min_confidence']:
                continue
                
            # Adjust coordinates if frame was resized
            if self.config['detector']['process_scale'] != 1.0:
                scale = 1.0 / self.config['detector']['process_scale']
                x1, x2 = int(x1 * scale), int(x2 * scale)
                y1, y2 = int(y1 * scale), int(y2 * scale)
            
            text = ' '.join(text) if isinstance(text, list) else text
            
            detections.append({
                'text': text,
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence
            })
        
        return detections

# [Continue with rest of detector class?]

    def process_video(self, source: str, output_path: Optional[str] = None) -> Dict:
        """
        Process video file or camera stream
        
        Args:
            source: Path to video file or camera index
            output_path: Path for output video file
            
        Returns:
            Dictionary containing processing statistics
        """
        try:
            # Open video source
            cap = self._open_video_source(source)
            video_info = self._get_video_properties(cap)
            writer = self._initialize_video_writer(output_path, video_info)
            
            logger.info(f"Processing video: {source}")
            logger.info(f"Video properties: {video_info['width']}x{video_info['height']} "
                       f"@ {video_info['fps']}fps")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame if it meets skip criterion
                if frame_count % self.config['detector']['frame_skip'] == 0:
                    processed_frame, detections = self.process_frame(frame)
                    
                    if writer:
                        writer.write(processed_frame)
                    
                    # Log detections
                    for det in detections:
                        if det.is_new:
                            logger.info(f"New plate detected: {det.text} "
                                      f"(confidence: {det.confidence:.2f})")
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
            
            # Cleanup and return statistics
            return self._cleanup_video_processing(cap, writer)
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
    
    def _open_video_source(self, source: str) -> cv2.VideoCapture:
        """Open video capture for file or camera"""
        try:
            if source.isdigit():
                cap = cv2.VideoCapture(int(source))
            else:
                cap = cv2.VideoCapture(source)
                
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {source}")
            return cap
        except Exception as e:
            logger.error(f"Failed to open video source: {str(e)}")
            raise

    def _get_video_properties(self, cap: cv2.VideoCapture) -> Dict:
        """Get video properties from capture object"""
        return {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

    def _initialize_video_writer(self, 
                               output_path: Optional[str], 
                               video_info: Dict) -> Optional[cv2.VideoWriter]:
        """Initialize video writer if output is enabled"""
        if not output_path:
            return None
            
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                video_info['fps'],
                (video_info['width'], video_info['height'])
            )
            logger.info(f"Initialized video writer: {output_path}")
            return writer
        except Exception as e:
            logger.error(f"Failed to initialize video writer: {str(e)}")
            return None

    def _cleanup_video_processing(self, 
                                cap: cv2.VideoCapture,
                                writer: Optional[cv2.VideoWriter]) -> Dict:
        """Cleanup video processing and return statistics"""
        try:
            cap.release()
            if writer:
                writer.release()
            
            stats = self.performance.get_stats()
            logger.info("Video processing complete:")
            logger.info(f"Processed {stats['frames_processed']} frames")
            logger.info(f"Average FPS: {stats['fps']:.1f}")
            logger.info(f"Total detections: {stats['total_detections']}")
            
            return stats
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise
        finally:
            cv2.destroyAllWindows()

    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process single image
        
        Args:
            image_path: Path to input image
            output_path: Path for output image
            
        Returns:
            Dictionary containing processing statistics
        """
        try:
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Process image
            processed_frame, detections = self.process_frame(frame)
            
            # Save output if path provided
            if output_path:
                cv2.imwrite(output_path, processed_frame)
                logger.info(f"Saved processed image to: {output_path}")
            
            # Log detections
            for det in detections:
                if det.is_new:
                    logger.info(f"Detected plate: {det.text} "
                              f"(confidence: {det.confidence:.2f})")
            
            return self.performance.get_stats()
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise