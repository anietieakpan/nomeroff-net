import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

class LoggerSetup:
    """Configure application logging"""
    
    @staticmethod
    def setup(config: dict) -> logging.Logger:
        """
        Set up application logger with file and console handlers
        
        Args:
            config: Logging configuration dictionary
        
        Returns:
            Configured logger instance
        """
        # Create logs directory if it doesn't exist
        log_path = Path(config['file']).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('license_plate_detector')
        logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            config['file'],
            maxBytes=config['max_size'],
            backupCount=config['backup_count']
        )
        file_handler.setLevel(getattr(logging, config['level']))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config['console_level']))
        console_formatter = logging.Formatter(
            '%(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

# Create a module-level logger
logger = logging.getLogger('license_plate_detector')