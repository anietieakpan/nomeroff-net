# src/following_pattern/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

class FollowingLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FollowingPattern:
    plate_number: str
    first_seen: datetime
    last_seen: datetime
    detection_count: int
    time_span_minutes: float
    detection_ratio: float
    average_confidence: float
    following_confidence: float
    following_level: FollowingLevel