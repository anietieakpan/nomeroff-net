import click
import yaml
from pathlib import Path
import cv2
from tqdm import tqdm
import logging

from .detector.plate_detector import LicensePlateDetector
from .detector.frame_processor import FrameProcessor
from .detector.visualization import DetectionVisualizer
from .tracking.detection_tracker import DetectionTracker
from .storage.sqlite_storage import SQLiteStorage
from .logging_config import setup_logging

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """License Plate Detection System"""
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--source', '-s', type=click.Path(exists=True),
              help='Path to video file or camera index')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory')
def detect(config, source, output):
    """Run license plate detection"""
    try:
        print(f"Starting detection with source: {source}")  # Debug print
        
        # Load configuration
        config_path = config or Path(__file__).parent.parent / 'config' / 'default_config.yaml'
        print(f"Using config from: {config_path}")  # Debug print
        
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Initialize components
        print("Initializing components...")  # Debug print
        frame_processor = FrameProcessor(min_confidence=cfg['detector']['min_confidence'])
        visualizer = DetectionVisualizer(cfg)
        tracker = DetectionTracker(max_persistence=cfg['detector']['detection_persistence'])
        storage = SQLiteStorage(cfg['storage']['path'])
        
        # Create detector
        detector = LicensePlateDetector(
            config=cfg,
            frame_processor=frame_processor,
            visualizer=visualizer,
            tracker=tracker,
            storage=storage
        )
        
        # Run detection
        print(f"Opening video source: {source}")  # Debug print
        if str(source).isdigit():
            print("Processing camera feed...")
            _process_camera(detector, int(source))
        else:
            print("Processing video file...")
            _process_video(detector, str(source))
            
    except Exception as e:
        print(f"Error in detection: {str(e)}")
        raise click.ClickException(str(e))

def _process_camera(detector: LicensePlateDetector, camera_id: int) -> None:
    """Process camera feed"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise click.ClickException(f"Could not open camera: {camera_id}")
        
    print("Camera opened successfully")  # Debug print
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from camera")  # Debug print
                break
                
            if frame_count == 0:
                print(f"First frame shape: {frame.shape}")  # Debug print
                
            visualization, success = detector.process_frame(frame)
            if success:
                cv2.imshow('License Plate Detection', visualization)
                if frame_count == 0:
                    print("First frame displayed")  # Debug print
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit command received")  # Debug print
                break
                
            frame_count += 1
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

def _process_video(detector: LicensePlateDetector, video_path: str) -> None:
    """Process video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise click.ClickException(f"Could not open video file: {video_path}")
        
    print("Video opened successfully")  # Debug print
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")  # Debug print
    
    try:
        with tqdm(total=total_frames) as pbar:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")  # Debug print
                    break
                
                if frame_count == 0:
                    print(f"First frame shape: {frame.shape}")  # Debug print
                    
                visualization, success = detector.process_frame(frame)
                if success:
                    cv2.imshow('License Plate Detection', visualization)
                    if frame_count == 0:
                        print("First frame displayed")  # Debug print
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit command received")  # Debug print
                    break
                
                frame_count += 1
                pbar.update(1)
                
        print(f"Processed {frame_count} frames")  # Debug print
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cli()