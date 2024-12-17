# src/following_analysis/reporting.py
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, Any
import json
import pandas as pd

class PatternReporter:
    """Generates reports from following pattern analysis"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        with sqlite3.connect(self.db_path) as conn:
            # Get overall statistics
            stats = self._get_overall_statistics(conn)
            
            # Get severity distribution
            severity_dist = self._get_severity_distribution(conn)
            
            # Get time-based patterns
            time_patterns = self._get_time_patterns(conn)
            
            # Get most frequent plates
            frequent_plates = self._get_frequent_plates(conn)
            
            return {
                'generated_at': datetime.now().isoformat(),
                'statistics': stats,
                'severity_distribution': severity_dist,
                'time_patterns': time_patterns,
                'frequent_plates': frequent_plates
            }

    def _get_overall_statistics(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Get overall statistics from analysis results"""
        cursor = conn.execute('''
            SELECT 
                COUNT(DISTINCT plate_number) as unique_plates,
                AVG(confidence_score) as avg_confidence,
                AVG(duration_minutes) as avg_duration,
                COUNT(*) as total_patterns
            FROM following_analysis
        ''')
        row = cursor.fetchone()
        return {
            'unique_plates': row[0],
            'average_confidence': row[1],
            'average_duration': row[2],
            'total_patterns': row[3]
        }

    def _get_severity_distribution(self, conn: sqlite3.Connection) -> Dict[str, int]:
        """Get distribution of severity levels"""
        cursor = conn.execute('''
            SELECT severity, COUNT(*) as count
            FROM following_analysis
            GROUP BY severity
        ''')
        return dict(cursor.fetchall())

    def _get_time_patterns(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Analyze patterns based on time of day"""
        df = pd.read_sql_query('''
            SELECT 
                time(first_seen) as detection_time,
                COUNT(*) as count
            FROM following_analysis
            GROUP BY strftime('%H', first_seen)
            ORDER BY detection_time
        ''', conn)
        
        return {
            'hourly_distribution': df.to_dict(orient='records'),
            'peak_hours': df.nlargest(3, 'count').to_dict(orient='records')
        }

    def _get_frequent_plates(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Get most frequently detected plates"""
        return pd.read_sql_query('''
            SELECT 
                plate_number,
                COUNT(*) as detection_count,
                AVG(confidence_score) as avg_confidence,
                MAX(severity) as highest_severity
            FROM following_analysis
            GROUP BY plate_number
            ORDER BY detection_count DESC
            LIMIT 10
        ''', conn).to_dict(orient='records')

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Save report to file"""
        with open(output_path, 'w') as f:
            if output_path.endswith('.json'):
                json.dump(report, f, indent=2)
            else:
                f.write(self.format_report(report))

    def format_report(self, report: Dict[str, Any]) -> str:
        """Format report as readable text"""
        lines = [
            "Following Vehicle Analysis Report",
            f"Generated: {report['generated_at']}",
            "\nOverall Statistics:",
            f"  Unique Plates: {report['statistics']['unique_plates']}",
            f"  Average Confidence: {report['statistics']['average_confidence']:.2f}",
            f"  Average Duration: {report['statistics']['average_duration']:.1f} minutes",
            "\nSeverity Distribution:"
        ]
        
        for severity, count in report['severity_distribution'].items():
            lines.append(f"  {severity}: {count}")
        
        lines.extend([
            "\nPeak Detection Hours:"
        ])
        
        for hour in report['time_patterns']['peak_hours']:
            lines.append(f"  {hour['detection_time']}: {hour['count']} detections")
        
        lines.extend([
            "\nMost Frequent Plates:"
        ])
        
        for plate in report['frequent_plates']:
            lines.append(
                f"  {plate['plate_number']}: "
                f"{plate['detection_count']} detections, "
                f"confidence: {plate['avg_confidence']:.2f}, "
                f"max severity: {plate['highest_severity']}"
            )
        
        return "\n".join(lines)