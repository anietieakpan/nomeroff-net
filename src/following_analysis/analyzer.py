# src/following_analysis/analyzer.py
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from .models import DetectionWindow, FollowingPattern, FollowingSeverity

logger = logging.getLogger(__name__)

class FollowingAnalyzer:
    """Analyzes vehicle following patterns from detection data"""
    
    def __init__(self, db_path: str, config: Dict[str, Any]):
        self.db_path = db_path
        self.config = self._validate_config(config)
        logger.info("Initialized FollowingAnalyzer")

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default configuration values"""
        defaults = {
            'min_detections': 5,
            'analysis_window_minutes': 30,
            'min_confidence': 0.6,
            'min_frequency': 0.1,  # detections per minute
            'severity_thresholds': {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'critical': 0.9
            }
        }
        
        # Update defaults with provided config
        defaults.update(config)
        return defaults

    def analyze_following_patterns(self) -> List[FollowingPattern]:
        """Analyze detection patterns to identify following vehicles"""
        try:
            # Get detection windows for analysis
            detection_windows = self._get_detection_windows()
            
            # Analyze each window for following patterns
            patterns = []
            for window in detection_windows:
                if self._meets_minimum_criteria(window):
                    pattern = self._analyze_detection_window(window)
                    if pattern:
                        patterns.append(pattern)
                        self._store_analysis_result(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing following patterns: {str(e)}")
            raise

    def _get_detection_windows(self) -> List[DetectionWindow]:
        """Retrieve detection windows for analysis"""
        cutoff_time = datetime.now() - timedelta(
            minutes=self.config['analysis_window_minutes']
        )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        plate_text,
                        MIN(timestamp) as first_seen,
                        MAX(timestamp) as last_seen,
                        COUNT(*) as detection_count,
                        AVG(confidence) as avg_confidence,
                        GROUP_CONCAT(timestamp) as detection_times
                    FROM detections
                    WHERE timestamp > ?
                    GROUP BY plate_text
                    HAVING COUNT(*) >= ?
                ''', (cutoff_time.isoformat(), self.config['min_detections']))
                
                windows = []
                for row in cursor.fetchall():
                    detection_times = sorted([
                        datetime.fromisoformat(ts)
                        for ts in row[5].split(',')
                    ])
                    
                    intervals = [
                        (detection_times[i] - detection_times[i-1]).total_seconds() / 60
                        for i in range(1, len(detection_times))
                    ]
                    
                    windows.append(DetectionWindow(
                        plate_number=row[0],
                        start_time=datetime.fromisoformat(row[1]),
                        end_time=datetime.fromisoformat(row[2]),
                        detection_count=row[3],
                        avg_confidence=row[4],
                        detection_intervals=intervals
                    ))
                
                return windows
                
        except sqlite3.Error as e:
            logger.error(f"Database error getting detection windows: {str(e)}")
            raise

    def _meets_minimum_criteria(self, window: DetectionWindow) -> bool:
        """Check if detection window meets minimum criteria for analysis"""
        duration = (window.end_time - window.start_time).total_seconds() / 60
        if duration == 0:
            return False
            
        frequency = window.detection_count / duration
        return (
            window.detection_count >= self.config['min_detections'] and
            window.avg_confidence >= self.config['min_confidence'] and
            frequency >= self.config['min_frequency']
        )

    def _analyze_detection_window(self, window: DetectionWindow) -> Optional[FollowingPattern]:
        """Analyze a detection window for following patterns"""
        try:
            # Calculate basic metrics
            duration = (window.end_time - window.start_time).total_seconds() / 60
            frequency = window.detection_count / duration
            
            # Analyze detection intervals
            interval_consistency = self._analyze_intervals(window.detection_intervals)
            
            # Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(
                window=window,
                frequency=frequency,
                interval_consistency=interval_consistency
            )
            
            # Determine severity
            severity = self._determine_severity(confidence_score)
            
            return FollowingPattern(
                plate_number=window.plate_number,
                first_seen=window.start_time,
                last_seen=window.end_time,
                total_detections=window.detection_count,
                duration_minutes=duration,
                detection_frequency=frequency,
                avg_confidence=window.avg_confidence,
                severity=severity,
                confidence_score=confidence_score,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing detection window: {str(e)}")
            return None

    def _analyze_intervals(self, intervals: List[float]) -> float:
        """Analyze consistency of detection intervals"""
        if not intervals:
            return 0.0
            
        # Calculate coefficient of variation (lower is more consistent)
        std_dev = np.std(intervals)
        mean = np.mean(intervals)
        if mean == 0:
            return 0.0
            
        cv = std_dev / mean
        # Convert to consistency score (0 to 1, higher is more consistent)
        return 1.0 / (1.0 + cv)

    def _calculate_confidence_score(self, window: DetectionWindow, 
                                 frequency: float, 
                                 interval_consistency: float) -> float:
        """Calculate confidence score for following pattern"""
        # Normalize factors
        duration_factor = min(
            (window.end_time - window.start_time).total_seconds() / 
            (self.config['analysis_window_minutes'] * 60), 
            1.0
        )
        frequency_factor = min(
            frequency / (self.config['min_frequency'] * 2),
            1.0
        )
        
        # Weighted combination
        confidence = (
            duration_factor * 0.3 +
            frequency_factor * 0.3 +
            interval_consistency * 0.2 +
            window.avg_confidence * 0.2
        )
        
        return min(confidence, 1.0)

    def _determine_severity(self, confidence_score: float) -> FollowingSeverity:
        """Determine severity level based on confidence score"""
        thresholds = self.config['severity_thresholds']
        
        if confidence_score >= thresholds['critical']:
            return FollowingSeverity.CRITICAL
        elif confidence_score >= thresholds['high']:
            return FollowingSeverity.HIGH
        elif confidence_score >= thresholds['medium']:
            return FollowingSeverity.MEDIUM
        elif confidence_score >= thresholds['low']:
            return FollowingSeverity.LOW
        return FollowingSeverity.NONE

    def _store_analysis_result(self, pattern: FollowingPattern) -> None:
        """Store analysis result in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create table if doesn't exist
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS following_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT NOT NULL,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        total_detections INTEGER NOT NULL,
                        duration_minutes REAL NOT NULL,
                        detection_frequency REAL NOT NULL,
                        avg_confidence REAL NOT NULL,
                        severity TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        analysis_timestamp TEXT NOT NULL
                    )
                ''')
                
                # Store result
                conn.execute('''
                    INSERT INTO following_analysis (
                        plate_number, first_seen, last_seen, total_detections,
                        duration_minutes, detection_frequency, avg_confidence,
                        severity, confidence_score, analysis_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.plate_number,
                    pattern.first_seen.isoformat(),
                    pattern.last_seen.isoformat(),
                    pattern.total_detections,
                    pattern.duration_minutes,
                    pattern.detection_frequency,
                    pattern.avg_confidence,
                    pattern.severity.value,
                    pattern.confidence_score,
                    pattern.analysis_timestamp.isoformat()
                ))
                
        except sqlite3.Error as e:
            logger.error(f"Error storing analysis result: {str(e)}")
            raise

# Example usage
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = {
        'min_detections': 5,
        'analysis_window_minutes': 30,
        'min_confidence': 0.6,
        'min_frequency': 0.1,
        'severity_thresholds': {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
    }
    
    analyzer = FollowingAnalyzer('analysis.db', config)
    patterns = analyzer.analyze_following_patterns()
    
    for pattern in patterns:
        if pattern.severity in [FollowingSeverity.HIGH, FollowingSeverity.CRITICAL]:
            print(f"\nPotential following vehicle detected:")
            print(f"Plate: {pattern.plate_number}")
            print(f"Duration: {pattern.duration_minutes:.1f} minutes")
            print(f"Detections: {pattern.total_detections}")
            print(f"Frequency: {pattern.detection_frequency:.2f} detections/minute")
            print(f"Confidence: {pattern.confidence_score:.2f}")
            print(f"Severity: {pattern.severity.value}")