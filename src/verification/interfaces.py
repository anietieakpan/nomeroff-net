# src/verification/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PlateVerificationResult:
    """Data class to hold verification results"""
    plate_number: str
    is_valid: bool
    verification_time: datetime
    source: str
    details: Dict[str, Any]
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None

class PlateVerificationSystem(ABC):
    """Abstract base class for plate verification systems"""
    
    @abstractmethod
    def verify_plate(self, plate_number: str) -> PlateVerificationResult:
        """Verify a single plate number"""
        pass
    
    @abstractmethod
    def verify_multiple(self, plate_numbers: list[str]) -> list[PlateVerificationResult]:
        """Verify multiple plate numbers"""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get current status of the verification system"""
        pass

