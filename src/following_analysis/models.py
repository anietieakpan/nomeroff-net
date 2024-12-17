# src/following_analysis/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum

class FollowingSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DetectionWindow:
    """Represents a time window of detections for a plate"""
    plate_number: str
    start_time: datetime
    end_time: datetime
    detection_count: int
    avg_confidence: float
    detection_intervals: List[float]  # intervals between detections in minutes

@dataclass
class FollowingPattern:
    """Represents an analyzed following pattern"""
    plate_number: str
    first_seen: datetime
    last_seen: datetime
    total_detections: int
    duration_minutes: float
    detection_frequency: float  # detections per minute
    avg_confidence: float
    severity: FollowingSeverity
    confidence_score: float
    analysis_timestamp: datetime

    