# src/db_sync/sync_service.py
import sqlite3
import logging
from datetime import datetime
import time
from typing import Optional
from pathlib import Path
from .models import SyncStatus, SyncConfig

logger = logging.getLogger(__name__)

class DatabaseSyncService:
    """Service responsible for real-time database synchronization"""
    
    def __init__(self, config: SyncConfig):
        self.config = config
        self._validate_config()
        self._init_target_db()
        self._last_sync_id = self._get_last_sync_id()
        logger.info(f"Initialized DB sync service from {config.source_db_path} to {config.target_db_path}")

    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not Path(self.config.source_db_path).exists():
            raise ValueError(f"Source database does not exist: {self.config.source_db_path}")
        
        if self.config.sync_interval < 1:
            raise ValueError("Sync interval must be at least 1 second")
        
        if self.config.batch_size < 1:
            raise ValueError("Batch size must be at least 1")

    def _init_target_db(self) -> None:
        """Initialize target database schema"""
        try:
            with sqlite3.connect(self.config.target_db_path) as conn:
                # Create detections table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        plate_text TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        x1 INTEGER, y1 INTEGER,
                        x2 INTEGER, y2 INTEGER,
                        source_type TEXT,
                        persistence_count INTEGER
                    )
                ''')
                
                # Create sync metadata table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS sync_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        last_sync_id INTEGER NOT NULL,
                        sync_timestamp TEXT NOT NULL,
                        records_synced INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        error_message TEXT
                    )
                ''')
                
                logger.info("Target database initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Error initializing target database: {str(e)}")
            raise

    def _get_last_sync_id(self) -> int:
        """Get the ID of the last synchronized record"""
        try:
            with sqlite3.connect(self.config.target_db_path) as conn:
                cursor = conn.execute('''
                    SELECT last_sync_id 
                    FROM sync_metadata 
                    ORDER BY sync_timestamp DESC 
                    LIMIT 1
                ''')
                result = cursor.fetchone()
                return result[0] if result else 0
        except sqlite3.Error as e:
            logger.error(f"Error getting last sync ID: {str(e)}")
            return 0

    def sync_once(self) -> Optional[SyncStatus]:
        """Perform a single synchronization operation"""
        try:
            with sqlite3.connect(self.config.source_db_path) as source_conn, \
                 sqlite3.connect(self.config.target_db_path) as target_conn:
                
                # Get new records from source
                cursor = source_conn.execute(f'''
                    SELECT * FROM detections 
                    WHERE id > ? 
                    ORDER BY id ASC 
                    LIMIT {self.config.batch_size}
                ''', (self._last_sync_id,))
                
                new_records = cursor.fetchall()
                if not new_records:
                    return None

                # Insert new records into target
                target_conn.executemany('''
                    INSERT OR REPLACE INTO detections 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', new_records)
                
                # Update sync metadata
                sync_time = datetime.now()
                last_id = new_records[-1][0]
                records_count = len(new_records)
                
                target_conn.execute('''
                    INSERT INTO sync_metadata 
                    (last_sync_id, sync_timestamp, records_synced, status)
                    VALUES (?, ?, ?, ?)
                ''', (last_id, sync_time.isoformat(), records_count, 'success'))
                
                target_conn.commit()
                self._last_sync_id = last_id
                
                status = SyncStatus(
                    last_sync_id=last_id,
                    last_sync_time=sync_time,
                    records_synced=records_count,
                    status='success'
                )
                
                logger.info(f"Synced {records_count} records successfully")
                return status
                
        except sqlite3.Error as e:
            error_msg = str(e)
            logger.error(f"Error during sync: {error_msg}")
            return SyncStatus(
                last_sync_id=self._last_sync_id,
                last_sync_time=datetime.now(),
                records_synced=0,
                status='error',
                error_message=error_msg
            )

    def start_sync_service(self) -> None:
        """Start continuous synchronization service"""
        logger.info("Starting continuous sync service")
        while True:
            try:
                status = self.sync_once()
                if status and status.status == 'error':
                    logger.error(f"Sync error: {status.error_message}")
                
                time.sleep(self.config.sync_interval)
                
            except KeyboardInterrupt:
                logger.info("Sync service stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in sync service: {str(e)}")
                time.sleep(self.config.sync_interval)

# Example usage
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sync configuration
    config = SyncConfig(
        source_db_path='license_plates.db',
        target_db_path='analysis.db',
        sync_interval=5,
        batch_size=1000
    )
    
    # Start sync service
    sync_service = DatabaseSyncService(config)
    sync_service.start_sync_service()

    # to run this independently: python -m src.db_sync.sync_service