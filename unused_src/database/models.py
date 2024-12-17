from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional

Base = declarative_base()

class Detection(Base):
    """SQLAlchemy model for license plate detections"""
    __tablename__ = 'detections'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    plate_text = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    x1 = Column(Integer, nullable=False)
    y1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    y2 = Column(Integer, nullable=False)
    source_type = Column(String, nullable=False)
    source_path = Column(String)
    persistence_count = Column(Integer, default=1)
    frame_number = Column(Integer)

    def __repr__(self):
        return (f"<Detection(plate='{self.plate_text}', "
                f"confidence={self.confidence:.2f}, "
                f"timestamp='{self.timestamp}')>")

class DetectionStats(Base):
    """SQLAlchemy model for detection statistics"""
    __tablename__ = 'detection_stats'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String, nullable=False, index=True)
    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    unique_plates = Column(Integer, default=0)
    avg_confidence = Column(Float)
    avg_processing_time = Column(Float)
    source_type = Column(String, nullable=False)
    source_path = Column(String)