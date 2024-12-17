import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import uuid

from .models import Base, Detection, DetectionStats

logger = logging.getLogger('license_plate_detector')

class DatabaseHandler:
    """Handle all database operations for license plate detection"""

    def __init__(self, db_url: str):
        """
        Initialize database connection
        
        Args:
            db_url: SQLAlchemy database URL
        """
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.session_id = str(uuid.uuid4())
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        logger.info(f"Initialized database connection: {db_url}")

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()

    def store_detection(self, 
                       plate_text: str,
                       confidence: float,
                       bbox: List[int],
                       source_type: str,
                       source_path: Optional[str] = None,
                       frame_number: Optional[int] = None,
                       persistence_count: int = 1) -> None:
        """
        Store a new detection in the database
        
        Args:
            plate_text: Detected license plate text
            confidence: Detection confidence score
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            source_type: Type of source (video/image/camera)
            source_path: Path to source file if applicable
            frame_number: Frame number for video sources
            persistence_count: Number of frames this plate has persisted
        """
        try:
            with self.session_scope() as session:
                detection = Detection(
                    plate_text=plate_text,
                    confidence=confidence,
                    x1=bbox[0],
                    y1=bbox[1],
                    x2=bbox[2],
                    y2=bbox[3],
                    source_type=source_type,
                    source_path=source_path,
                    frame_number=frame_number,
                    persistence_count=persistence_count
                )
                session.add(detection)
                logger.debug(f"Stored detection: {plate_text} ({confidence:.2f})")
        except Exception as e:
            logger.error(f"Failed to store detection: {str(e)}")
            raise

    def get_recent_detections(self, 
                            limit: int = 10, 
                            minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Get recent detections from the database
        
        Args:
            limit: Maximum number of detections to return
            minutes: Time window in minutes
            
        Returns:
            List of detection dictionaries
        """
        try:
            with self.session_scope() as session:
                cutoff = datetime.utcnow() - timedelta(minutes=minutes)
                detections = (session.query(Detection)
                            .filter(Detection.timestamp >= cutoff)
                            .order_by(Detection.timestamp.desc())
                            .limit(limit)
                            .all())
                
                return [{
                    'plate_text': d.plate_text,
                    'confidence': d.confidence,
                    'timestamp': d.timestamp,
                    'bbox': [d.x1, d.y1, d.x2, d.y2],
                    'persistence_count': d.persistence_count
                } for d in detections]
        except Exception as e:
            logger.error(f"Failed to get recent detections: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics from the database
        
        Returns:
            Dictionary containing detection statistics
        """
        try:
            with self.session_scope() as session:
                # Get basic statistics
                stats = session.query(
                    func.count(Detection.id).label('total_detections'),
                    func.count(func.distinct(Detection.plate_text)).label('unique_plates'),
                    func.avg(Detection.confidence).label('avg_confidence'),
                    func.avg(Detection.persistence_count).label('avg_persistence')
                ).first()
                
                return {
                    'total_detections': stats[0],
                    'unique_plates': stats[1],
                    'average_confidence': float(stats[2]) if stats[2] else 0,
                    'average_persistence': float(stats[3]) if stats[3] else 0
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            raise

    def cleanup_old_detections(self, days: int = 30) -> int:
        """
        Remove detections older than specified days
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of records deleted
        """
        try:
            with self.session_scope() as session:
                cutoff = datetime.utcnow() - timedelta(days=days)
                deleted = (session.query(Detection)
                         .filter(Detection.timestamp < cutoff)
                         .delete())
                logger.info(f"Cleaned up {deleted} old detections")
                return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old detections: {str(e)}")
            raise

    def store_session_stats(self, stats: Dict[str, Any]) -> None:
        """
        Store statistics for the current detection session
        
        Args:
            stats: Dictionary containing session statistics
        """
        try:
            with self.session_scope() as session:
                session_stats = DetectionStats(
                    session_id=self.session_id,
                    total_frames=stats.get('total_frames', 0),
                    processed_frames=stats.get('processed_frames', 0),
                    total_detections=stats.get('total_detections', 0),
                    unique_plates=stats.get('unique_plates', 0),
                    avg_confidence=stats.get('avg_confidence', 0),
                    avg_processing_time=stats.get('avg_process_time', 0),
                    source_type=stats.get('source_type', 'unknown'),
                    source_path=stats.get('source_path')
                )
                session.add(session_stats)
                logger.info(f"Stored session statistics for {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to store session stats: {str(e)}")
            raise