# tests/conftest.py
import pytest
import tempfile
import yaml
from pathlib import Path
from src.storage.sqlite_storage import SQLiteStorage

@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix='.db') as f:
        yield f.name

@pytest.fixture
def sqlite_storage(temp_db_path):
    storage = SQLiteStorage(temp_db_path)
    yield storage
    storage.cleanup()

@pytest.fixture
def sample_detection():
    return {
        'text': 'ABC123',
        'confidence': 0.95,
        'bbox': [100, 100, 200, 150],
        'image_path': 'test_image.jpg',
        'source_type': 'image'
    }