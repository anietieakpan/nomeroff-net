import os
from pathlib import Path
from typing import Dict, Any
import yaml
import logging

logger = logging.getLogger('license_plate_detector')

class ConfigLoader:
    """Handle loading and validation of configuration"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize config loader
        
        Args:
            config_path: Path to custom config file (optional)
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = {}
    
    @staticmethod
    def _get_default_config_path() -> str:
        """Get path to default config file"""
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config',
            'default_config.yaml'
        )
    
    def load(self) -> Dict[str, Any]:
        """
        Load and validate configuration
        
        Returns:
            Validated configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self._validate_config()
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {str(e)}")
            raise
    
    def _validate_config(self):
        """Validate required configuration parameters"""
        required_sections = ['detector', 'visualization', 'database', 'logging', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate specific required parameters
        if 'level' not in self.config['logging']:
            raise ValueError("Missing required logging level")
        
        if 'url' not in self.config['database']:
            raise ValueError("Missing required database URL")
    
    def update_from_env(self):
        """Update configuration from environment variables"""
        env_mappings = {
            'DETECTOR_CONFIDENCE': ('detector', 'min_confidence', float),
            'DB_URL': ('database', 'url', str),
            'LOG_LEVEL': ('logging', 'level', str),
            'DEBUG_MODE': ('visualization', 'debug_mode', lambda x: x.lower() == 'true')
        }
        
        for env_var, (section, key, type_conv) in env_mappings.items():
            if env_var in os.environ:
                try:
                    self.config[section][key] = type_conv(os.environ[env_var])
                except Exception as e:
                    logger.warning(
                        f"Failed to set {env_var} config value: {str(e)}"
                    )