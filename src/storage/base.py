from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime

class DetectionStorage(ABC):
    """Abstract base class for detection storage"""
    
    @abstractmethod
    def store_detection(self, plate_text: str, confidence: float, 
                       bbox: List[int], image_path: str, 
                       source_type: str, persistence_count: int = 1) -> None:
        """Store a single detection"""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        pass