import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import sys

logger = logging.getLogger('license_plate_detector')

class CLIParser:
    """Handle command line interface for the license plate detector"""
    
    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser"""
        parser = argparse.ArgumentParser(
            description='License Plate Detection System',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Mode selection
        parser.add_argument(
            '--mode',
            type=str,
            choices=['image', 'video', 'camera'],
            required=True,
            help='Processing mode'
        )

        # Source specification
        parser.add_argument(
            '--source',
            type=str,
            help='Path to image/video file or camera index'
        )

        # Configuration file
        parser.add_argument(
            '--config',
            type=str,
            help='Path to custom configuration file'
        )

        # Processing options
        processing_group = parser.add_argument_group('Processing Options')
        processing_group.add_argument(
            '--frame-skip',
            type=int,
            help='Process every nth frame'
        )
        processing_group.add_argument(
            '--confidence',
            type=float,
            help='Minimum confidence threshold (0-1)'
        )
        processing_group.add_argument(
            '--persistence',
            type=int,
            help='Number of frames to persist detections'
        )

        # Display options
        display_group = parser.add_argument_group('Display Options')
        display_group.add_argument(
            '--display-scale',
            type=float,
            help='Display window scale (0.1-1.0)'
        )
        display_group.add_argument(
            '--no-display',
            action='store_true',
            help='Disable visualization'
        )
        display_group.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug mode'
        )

        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument(
            '--output',
            type=str,
            help='Output path for processed video/image'
        )
        output_group.add_argument(
            '--no-save',
            action='store_true',
            help='Disable saving output files'
        )

        # Database options
        db_group = parser.add_argument_group('Database Options')
        db_group.add_argument(
            '--db-path',
            type=str,
            help='Custom database path'
        )
        db_group.add_argument(
            '--list-plates',
            action='store_true',
            help='List all detected plates and exit'
        )
        db_group.add_argument(
            '--clear-old',
            type=int,
            help='Clear detections older than specified days'
        )

        return parser

    def parse_args(self) -> Dict[str, Any]:
        """
        Parse and validate command line arguments
        
        Returns:
            Dictionary of validated arguments
        """
        args = self.parser.parse_args()
        
        try:
            self._validate_args(args)
            return self._process_args(args)
        except ValueError as e:
            self.parser.error(str(e))

    def _validate_args(self, args):
        """Validate command line arguments"""
        # Check source requirement
        if args.mode in ['image', 'video'] and not args.source:
            raise ValueError(f"{args.mode} mode requires --source argument")

        # Validate numeric ranges
        if args.frame_skip is not None and args.frame_skip < 1:
            raise ValueError("Frame skip must be >= 1")

        if args.confidence is not None and not 0 <= args.confidence <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        if args.display_scale is not None and not 0.1 <= args.display_scale <= 1.0:
            raise ValueError("Display scale must be between 0.1 and 1.0")

        # Check file existence
        if args.config and not Path(args.config).is_file():
            raise ValueError(f"Config file not found: {args.config}")

        if args.mode in ['image', 'video'] and not Path(args.source).is_file():
            raise ValueError(f"Source file not found: {args.source}")

    def _process_args(self, args) -> Dict[str, Any]:
        """Process and structure arguments"""
        # Convert arguments to configuration dictionary
        config = {
            'mode': args.mode,
            'source': args.source,
            'detector': {
                'frame_skip': args.frame_skip,
                'min_confidence': args.confidence,
                'detection_persistence': args.persistence
            },
            'visualization': {
                'display_scale': args.display_scale,
                'show_display': not args.no_display,
                'debug_mode': args.debug
            },
            'output': {
                'path': args.output,
                'save_output': not args.no_save
            },
            'database': {
                'path': args.db_path,
                'list_plates': args.list_plates,
                'clear_old_days': args.clear_old
            }
        }

        # Remove None values
        return {k: v for k, v in config.items() if v is not None}

def main():
    """CLI entry point"""
    try:
        cli = CLIParser()
        config = cli.parse_args()
        
        # Initialize and run detector
        from core.detector import LicensePlateDetector
        detector = LicensePlateDetector(config)
        
        if config['database'].get('list_plates'):
            # Just list plates and exit
            detector.list_stored_plates()
            sys.exit(0)
            
        if config['database'].get('clear_old_days'):
            # Clear old detections and exit
            detector.clear_old_detections(config['database']['clear_old_days'])
            sys.exit(0)
            
        # Run detection based on mode
        if config['mode'] == 'image':
            detector.process_image(config['source'], config['output'].get('path'))
        else:
            detector.process_video(config['source'], config['output'].get('path'))
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()