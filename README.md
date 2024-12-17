# License Plate Detection and Following Vehicle Analysis System

A sophisticated system that combines real-time license plate detection with vehicle following pattern analysis. The system captures license plates using computer vision and analyzes temporal patterns to identify potential following vehicles.

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Database Schema](#database-schema)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Features
- Real-time license plate detection using Nomeroff-net
- Automatic database synchronization
- Following vehicle pattern analysis
- Statistical reporting and visualization
- Command-line interface for analysis
- Configurable detection and analysis parameters

## System Architecture

### Components
```
[License Plate Detection] → [Primary DB] → [DB Sync Service] → [Analysis DB] → [Following Analysis]
```

### Project Structure
```
license_plate_detector/
├── src/
│   ├── detector/           # License plate detection
│   │   ├── plate_detector.py
│   │   └── visualization.py
│   ├── db_sync/           # Database synchronization
│   │   ├── models.py
│   │   └── sync_service.py
│   ├── following_analysis/ # Following pattern analysis
│   │   ├── analyzer.py
│   │   ├── models.py
│   │   ├── visualization.py
│   │   ├── reporting.py
│   │   └── cli.py
│   └── utils/
│       └── config.py
├── config/
│   └── config.yaml        # Configuration file
├── tests/                 # Unit tests
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- SQLite3

### Setup
1. Clone the repository
```bash
git clone <repository-url>
cd license_plate_detector
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install requirements
```bash
pip install -r requirements.txt
```

## Database Schema

### Primary Database (license_plates.db)
```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    plate_text TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 INTEGER, y1 INTEGER,
    x2 INTEGER, y2 INTEGER,
    source_type TEXT,
    persistence_count INTEGER
);
```

### Analysis Database (analysis.db)
```sql
-- Synchronized detections table
CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    plate_text TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 INTEGER, y1 INTEGER,
    x2 INTEGER, y2 INTEGER,
    source_type TEXT,
    persistence_count INTEGER
);

-- Following analysis results
CREATE TABLE following_analysis (
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
);
```

## Usage

### License Plate Detection
```bash
# Process video file
python -m src.cli detect --source video.mp4

# Use webcam
python -m src.cli detect --source 0
```

### Database Synchronization
```bash
# Start sync service
python -m src.db_sync.sync_service
```

### Following Analysis
```bash
# Analyze patterns
python -m src.following_analysis.cli analyze --db analysis.db --window 30

# Generate report
python -m src.following_analysis.cli report --db analysis.db --output report.txt
```

### Pattern Visualization
```python
from src.following_analysis.visualization import PatternVisualizer

visualizer = PatternVisualizer()
visualizer.plot_detection_timeline(patterns, 'timeline.png')
visualizer.plot_detection_heatmap(patterns, 'heatmap.png')
```

## Configuration

Configuration is managed through `config/config.yaml`:

```yaml
plate_detection:
  min_confidence: 0.5    # Minimum confidence for plate detection
  frame_skip: 2          # Process every nth frame

db_sync:
  sync_interval: 5       # Seconds between syncs
  batch_size: 1000      # Records per sync batch
  max_retries: 3        # Max retry attempts

following_detection:
  min_detections: 5              # Minimum detections to consider
  time_window_minutes: 30        # Analysis time window
  min_detection_ratio: 0.3       # Minimum detections per minute
  analysis_interval: 300         # Seconds between analyses
  severity_thresholds:
    low: 0.3
    medium: 0.5
    high: 0.7
    critical: 0.9
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Adding New Features
1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Keep functions focused

## Troubleshooting

### Common Issues

1. **Database Errors**
   - Check file permissions
   - Verify database paths
   - Check disk space

2. **Detection Issues**
   - Verify camera connection
   - Check lighting conditions
   - Adjust confidence threshold

3. **Sync Issues**
   - Check network connectivity
   - Verify database paths
   - Check sync interval settings

### Logging
Logs are stored in:
- `license_plate_detector.log` - Main application
- `db_sync.log` - Sync service
- `following_analysis.log` - Analysis service

### Performance Tips
1. Adjust frame_skip for optimal performance
2. Configure appropriate sync intervals
3. Monitor system resources
4. Regular database maintenance

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Submit pull request

## License
[Insert License Information]

## Contact
[Insert Contact Information]


