# src/following_pattern/analyzer.py
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
from .models import FollowingPattern, FollowingLevel

logger = logging.getLogger(__name__)

class FollowingPatternAnalyzer:
    """Analyzes time-series license plate detection data for following patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_detections = config.get('min_detections', 5)
        self.time_window_minutes = config.get('time_window', 30)
        self.min_detection_ratio = config.get('min_detection_ratio', 0.3)
        logger.info("Initialized FollowingPatternAnalyzer")

    def analyze_patterns(self, db_path: str) -> List[FollowingPattern]:
        """Analyze database for following patterns"""
        try:
            with sqlite3.connect(db_path) as conn:
                patterns = self._query_detection_patterns(conn)
                analyzed_patterns = [
                    self._analyze_single_pattern(data)
                    for data in patterns if self._meets_minimum_criteria(data)
                ]
                return [p for p in analyzed_patterns if p is not None]
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return []

    def _query_detection_patterns(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Query detection patterns from database"""
        cutoff_time = datetime.now() - timedelta(minutes=self.time_window_minutes)
        cursor = conn.execute('''
            SELECT 
                plate_text,
                COUNT(*) as detection_count,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen,
                AVG(confidence) as avg_confidence,
                GROUP_CONCAT(timestamp) as detection_times
            FROM detections
            WHERE timestamp > ?
            GROUP BY plate_text
            HAVING COUNT(*) >= ?
        ''', (cutoff_time.isoformat(), self.min_detections))
        
        return [dict(row) for row in cursor.fetchall()]

    def _meets_minimum_criteria(self, data: Dict[str, Any]) -> bool:
        """Check if detection pattern meets minimum criteria"""
        try:
            time_span = (
                datetime.fromisoformat(data['last_seen']) - 
                datetime.fromisoformat(data['first_seen'])
            ).total_seconds() / 60
            
            detection_ratio = data['detection_count'] / time_span if time_span > 0 else 0
            return detection_ratio >= self.min_detection_ratio
        except Exception as e:
            logger.error(f"Error checking criteria: {str(e)}")
            return False

    def _analyze_single_pattern(self, data: Dict[str, Any]) -> Optional[FollowingPattern]:
        """Analyze a single plate's detection pattern"""
        try:
            time_span = (
                datetime.fromisoformat(data['last_seen']) - 
                datetime.fromisoformat(data['first_seen'])
            ).total_seconds() / 60
            
            detection_ratio = data['detection_count'] / time_span if time_span > 0 else 0
            following_confidence = self._calculate_following_confidence(
                detection_ratio, time_span, data['detection_count'], data['avg_confidence']
            )
            
            return FollowingPattern(
                plate_number=data['plate_text'],
                first_seen=datetime.fromisoformat(data['first_seen']),
                last_seen=datetime.fromisoformat(data['last_seen']),
                detection_count=data['detection_count'],
                time_span_minutes=time_span,
                detection_ratio=detection_ratio,
                average_confidence=data['avg_confidence'],
                following_confidence=following_confidence,
                following_level=self._determine_following_level(following_confidence)
            )
        except Exception as e:
            logger.error(f"Error analyzing pattern: {str(e)}")
            return None

    def _calculate_following_confidence(self, detection_ratio: float, 
                                     time_span: float, count: int, 
                                     avg_confidence: float) -> float:
        """Calculate confidence score that vehicle is following"""
        try:
            # Normalize factors
            time_factor = min(time_span / self.time_window_minutes, 1.0)
            detection_factor = min(detection_ratio / self.min_detection_ratio, 1.0)
            count_factor = min(count / (self.min_detections * 2), 1.0)
            
            # Weighted average
            weights = self.config.get('confidence_weights', {
                'time_span': 0.4,
                'detection_ratio': 0.4,
                'count': 0.1,
                'confidence': 0.1
            })
            
            confidence = (
                time_factor * weights['time_span'] +
                detection_factor * weights['detection_ratio'] +
                count_factor * weights['count'] +
                avg_confidence * weights['confidence']
            )
            
            return min(confidence, 1.0)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def _determine_following_level(self, confidence: float) -> FollowingLevel:
        """Determine following level based on confidence score"""
        if confidence >= 0.9:
            return FollowingLevel.CRITICAL
        elif confidence >= 0.7:
            return FollowingLevel.HIGH
        elif confidence >= 0.5:
            return FollowingLevel.MEDIUM
        elif confidence >= 0.3:
            return FollowingLevel.LOW
        return FollowingLevel.NONE
