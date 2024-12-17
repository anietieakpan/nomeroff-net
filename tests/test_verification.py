# tests/test_verification.py
import pytest
from datetime import datetime
from src.verification.verifier import PlateVerifier, MockVerificationSystem
from src.verification.interfaces import PlateVerificationResult

@pytest.fixture
def verifier(temp_db_path):
    return PlateVerifier(temp_db_path, MockVerificationSystem())

def test_verify_plate(verifier):
    # Test verifying a single plate
    result = verifier.verify_plate_from_db("ABC123")
    assert isinstance(result, PlateVerificationResult)
    assert result.plate_number == "ABC123"
    assert result.is_valid is True

def test_verify_multiple(verifier):
    # Test batch verification
    results = verifier.verify_all_unverified()
    assert isinstance(results, list)
    for result in results:
        assert isinstance(result, PlateVerificationResult)