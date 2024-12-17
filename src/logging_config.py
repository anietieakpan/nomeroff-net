# src/logging_config.py
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path: str = None):
    """Setup logging configuration"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'default_config.yaml'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config['logging'])

    return logging.getLogger(__name__)