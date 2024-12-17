# tests/test_following_analysis.py
import pytest
from datetime import datetime, timedelta
import sqlite3
import tempfile
from src.following_analysis.analyzer import FollowingAnalyzer
from src.following_analysis.models import DetectionWindow, FollowingPattern, FollowingSeverity

@pytest.fixture
def test_db():
    """Create temporary test database"""
    with tempfile.NamedTemporaryFile() as f:
        conn = sqlite3.connect(f.name)
        conn.execute('''
            CREATE TABLE detections (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                plate_text TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        ''')
        return f.name

@pytest.fixture
def analyzer(test_db):
    config = {
        'min_detections': 3,
        'analysis_window_minutes': 15,
        'min_confidence': 0.5
    }
    return FollowingAnalyzer(test_db, config)

def test_following_pattern_detection(analyzer, test_db):
    # Insert test data
    with sqlite3.connect(test_db) as conn:
        base_time = datetime.now()
        test_data = [
            ('ABC123', base_time + timedelta(minutes=i*5), 0.95)
            for i in range(5)
        ]
        conn.executemany(
            'INSERT INTO detections (plate_text, timestamp, confidence) VALUES (?, ?, ?)',
            [(plate, time.isoformat(), conf) for plate, time, conf in test_data]
        )
    
    patterns = analyzer.analyze_following_patterns()
    assert len(patterns) > 0
    pattern = patterns[0]
    assert pattern.plate_number == 'ABC123'
    assert pattern.total_detections == 5
    assert pattern.confidence_score > 0.5

def test_severity_classification(analyzer):
    window = DetectionWindow(
        plate_number='XYZ789',
        start_time=datetime.now() - timedelta(minutes=20),
        end_time=datetime.now(),
        detection_count=10,
        avg_confidence=0.9,
        detection_intervals=[5.0] * 9  # Consistent 5-minute intervals
    )
    
    pattern = analyzer._analyze_detection_window(window)
    assert pattern is not None
    assert pattern.severity in [FollowingSeverity.HIGH, FollowingSeverity.CRITICAL]