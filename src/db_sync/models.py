# src/db_sync/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class SyncStatus:
    last_sync_id: int
    last_sync_time: datetime
    records_synced: int
    status: str
    error_message: Optional[str] = None

@dataclass
class SyncConfig:
    source_db_path: str
    target_db_path: str
    sync_interval: int = 5  # seconds
    batch_size: int = 1000
    max_retries: int = 3