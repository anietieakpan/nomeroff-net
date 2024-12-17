# src/following_analysis/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
from typing import List
from .models import FollowingPattern

class PatternVisualizer:
    """Visualizes following pattern analysis results"""
    
    def plot_detection_timeline(self, patterns: List[FollowingPattern], 
                              output_path: str = None):
        """Create timeline visualization of detections"""
        # Prepare data
        data = []
        for pattern in patterns:
            data.append({
                'plate': pattern.plate_number,
                'start': pattern.first_seen,
                'end': pattern.last_seen,
                'severity': pattern.severity.value,
                'confidence': pattern.confidence_score
            })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot timeline
        sns.scatterplot(
            data=df,
            x='start',
            y='plate',
            hue='severity',
            size='confidence',
            sizes=(50, 200)
        )
        
        plt.title('Vehicle Following Pattern Timeline')
        plt.xlabel('Time')
        plt.ylabel('License Plate')
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def plot_detection_heatmap(self, patterns: List[FollowingPattern], 
                             output_path: str = None):
        """Create heatmap of detection frequencies"""
        # Prepare data
        data = {
            'plate': [],
            'hour': [],
            'frequency': []
        }
        
        for pattern in patterns:
            hour = pattern.first_seen.hour
            data['plate'].append(pattern.plate_number)
            data['hour'].append(hour)
            data['frequency'].append(pattern.detection_frequency)
        
        df = pd.DataFrame(data)
        pivot_data = df.pivot('plate', 'hour', 'frequency')
        
        # Create heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            pivot_data,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f'
        )
        
        plt.title('Detection Frequency Heatmap')
        plt.xlabel('Hour of Day')
        plt.ylabel('License Plate')
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()