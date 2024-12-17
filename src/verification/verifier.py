# src/verification/verifier.py
import logging
from typing import Optional, List, Dict, Any
import sqlite3
from datetime import datetime
from .interfaces import PlateVerificationSystem, PlateVerificationResult

logger = logging.getLogger(__name__)

class PlateVerifier:
    """Main class for handling license plate verification"""
    
    def __init__(self, db_path: str, verification_system: PlateVerificationSystem):
        self.db_path = db_path
        self.verification_system = verification_system
        logger.info(f"Initialized PlateVerifier with database: {db_path}")
    
    def verify_plate_from_db(self, plate_number: str) -> Optional[PlateVerificationResult]:
        """Verify a specific plate number from the database"""
        try:
            # First check if plate exists in our database
            if not self._plate_exists_in_db(plate_number):
                logger.warning(f"Plate number {plate_number} not found in database")
                return None
            
            # Verify with external system
            result = self.verification_system.verify_plate(plate_number)
            
            # Store verification result
            self._store_verification_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error verifying plate {plate_number}: {str(e)}")
            raise
    
    def verify_all_unverified(self) -> List[PlateVerificationResult]:
        """Verify all unverified plates in the database"""
        try:
            unverified_plates = self._get_unverified_plates()
            results = []
            
            for plate in unverified_plates:
                try:
                    result = self.verify_plate_from_db(plate)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error verifying plate {plate}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch verification: {str(e)}")
            raise
    
    def _plate_exists_in_db(self, plate_number: str) -> bool:
        """Check if a plate number exists in the detection database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM detections WHERE plate_text = ?',
                    (plate_number,)
                )
                return cursor.fetchone()[0] > 0
        except sqlite3.Error as e:
            logger.error(f"Database error checking plate existence: {str(e)}")
            raise
    
    def _get_unverified_plates(self) -> List[str]:
        """Get list of plates that haven't been verified yet"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT DISTINCT d.plate_text 
                    FROM detections d 
                    LEFT JOIN verifications v ON d.plate_text = v.plate_number 
                    WHERE v.plate_number IS NULL
                ''')
                return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error getting unverified plates: {str(e)}")
            raise
    
    def _store_verification_result(self, result: PlateVerificationResult) -> None:
        """Store verification result in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First, ensure we have a verifications table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS verifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT NOT NULL,
                        verification_time TIMESTAMP NOT NULL,
                        is_valid BOOLEAN NOT NULL,
                        source TEXT NOT NULL,
                        confidence_score REAL,
                        details TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Store the verification result
                conn.execute('''
                    INSERT INTO verifications 
                    (plate_number, verification_time, is_valid, source, 
                     confidence_score, details, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.plate_number,
                    result.verification_time.isoformat(),
                    result.is_valid,
                    result.source,
                    result.confidence_score,
                    str(result.details),
                    str(result.metadata) if result.metadata else None
                ))
                conn.commit()
                logger.debug(f"Stored verification result for plate {result.plate_number}")
        except sqlite3.Error as e:
            logger.error(f"Database error storing verification result: {str(e)}")
            raise


# Example implementation of a verification system
class MockVerificationSystem(PlateVerificationSystem):
    """Mock implementation for testing purposes"""
    
    def verify_plate(self, plate_number: str) -> PlateVerificationResult:
        """Mock verification of a single plate"""
        # Simulate external system verification
        return PlateVerificationResult(
            plate_number=plate_number,
            is_valid=True,  # Mock result
            verification_time=datetime.now(),
            source="mock_system",
            details={"check_type": "mock"},
            confidence_score=0.95
        )
    
    def verify_multiple(self, plate_numbers: list[str]) -> list[PlateVerificationResult]:
        """Mock verification of multiple plates"""
        return [self.verify_plate(plate) for plate in plate_numbers]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get mock system status"""
        return {
            "status": "operational",
            "latency": "10ms",
            "api_version": "1.0"
        }