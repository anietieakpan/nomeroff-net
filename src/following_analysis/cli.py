# src/following_analysis/cli.py
import click
import yaml
from datetime import datetime, timedelta
from .analyzer import FollowingAnalyzer
from .visualization import PatternVisualizer
from .reporting import PatternReporter

@click.group()
def cli():
    """Following Vehicle Analysis CLI"""
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--db', '-d', type=click.Path(exists=True), required=True, help='Analysis database path')
@click.option('--window', '-w', type=int, default=30, help='Analysis window in minutes')
def analyze(config, db, window):
    """Analyze following patterns"""
    # Load configuration
    if config:
        with open(config) as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {}
    
    # Update window if specified
    config_data['analysis_window_minutes'] = window
    
    # Run analysis
    analyzer = FollowingAnalyzer(db, config_data)
    patterns = analyzer.analyze_following_patterns()
    
    # Print results
    click.echo(f"\nAnalyzed patterns in {window} minute window:")
    for pattern in patterns:
        click.echo(
            f"\nPlate: {pattern.plate_number}\n"
            f"Severity: {pattern.severity.value}\n"
            f"Confidence: {pattern.confidence_score:.2f}\n"
            f"Duration: {pattern.duration_minutes:.1f} minutes\n"
            f"Detections: {pattern.total_detections}"
        )

@cli.command()
@click.option('--db', '-d', type=click.Path(exists=True), required=True, help='Analysis database path')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def report(db, output):
    """Generate analysis report"""
    reporter = PatternReporter(db)
    report_data = reporter.generate_report()
    
    if output:
        reporter.save_report(report_data, output)
        click.echo(f"Report saved to {output}")
    else:
        click.echo(reporter.format_report(report_data))