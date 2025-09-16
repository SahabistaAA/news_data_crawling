import psycopg2 as pg2
import pandas as pd
from psycopg2.extras import execute_values, RealDictCursor
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class BronzeDB:
    """Database operations for Bronze layer"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'airflow'),
            'user': os.getenv('DB_USER', 'airflow'),
            'password': os.getenv('DB_PASSWORD', 'airflow_11')
        }
        
        self.raw_table = 'raw_data'
        self.bronze_table = 'bronze_data'
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = pg2.connect(**self.db_config)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def create_bronze_table(self) -> None:
        """Create bronze table if it doesn't exist"""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.bronze_table} (
            id SERIAL PRIMARY KEY,
            title TEXT,
            url TEXT,
            pub_date TIMESTAMP,
            source TEXT,
            region TEXT,
            author TEXT,
            content TEXT,
            full_content TEXT,
            ingested_at TIMESTAMP,
            cleaned_at TIMESTAMP,
            is_duplicate BOOLEAN DEFAULT FALSE,
            quality_check_status VARCHAR(50),
            language_detected VARCHAR(10),
            content_type VARCHAR(50),
            title_length INTEGER,
            content_length INTEGER,
            full_content_length INTEGER,
            content_truncated BOOLEAN DEFAULT FALSE,
            title_truncated BOOLEAN DEFAULT FALSE,
            title_word_count INTEGER,
            content_word_count INTEGER,
            readability_score FLOAT,
            quality_score FLOAT,
            domain VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_bronze_url ON {self.bronze_table}(url);
        CREATE INDEX IF NOT EXISTS idx_bronze_source ON {self.bronze_table}(source);
        CREATE INDEX IF NOT EXISTS idx_bronze_region ON {self.bronze_table}(region);
        CREATE INDEX IF NOT EXISTS idx_bronze_pub_date ON {self.bronze_table}(pub_date);
        CREATE INDEX IF NOT EXISTS idx_bronze_quality ON {self.bronze_table}(quality_check_status);
        CREATE INDEX IF NOT EXISTS idx_bronze_language ON {self.bronze_table}(language_detected);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_query)
                    conn.commit()
            logger.info(f"Bronze table '{self.bronze_table}' created/verified successfully")
        except Exception as e:
            logger.error(f"Error creating bronze table: {e}")
            raise
    
    def fetch_raw_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch raw data from raw_data table"""
        try:
            query = f"""
            SELECT 
                title, url, pub_date, source, region, author, 
                content, full_content, ingested_at
            FROM {self.raw_table}
            WHERE 1=1
            ORDER BY ingested_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn)
            
            logger.info(f"Fetched {len(df)} raw records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching raw data: {e}")
            raise
    
    def save_to_bronze(self, df: pd.DataFrame) -> None:
        """Save processed data to bronze table"""
        if df.empty:
            logger.warning("No data to save to bronze table")
            return
        
        try:
            # Prepare column list for insertion
            columns = [
                'title', 'url', 'pub_date', 'source', 'region', 'author',
                'content', 'full_content', 'ingested_at', 'cleaned_at',
                'is_duplicate', 'quality_check_status', 'language_detected',
                'content_type', 'title_length', 'content_length',
                'full_content_length', 'content_truncated', 'title_truncated',
                'title_word_count', 'content_word_count', 'readability_score',
                'quality_score', 'domain'
            ]
            
            # Ensure all columns exist in DataFrame
            for col in columns:
                if col not in df.columns:
                    if col in ['cleaned_at', 'ingested_at']:
                        df[col] = pd.Timestamp.now()
                    elif col in ['is_duplicate', 'content_truncated', 'title_truncated']:
                        df[col] = False
                    elif col in ['title_length', 'content_length', 'full_content_length',
                               'title_word_count', 'content_word_count']:
                        df[col] = 0
                    elif col in ['readability_score', 'quality_score']:
                        df[col] = 0.0
                    else:
                        df[col] = ''
            
            # Prepare data for insertion
            data_tuples = []
            for _, row in df.iterrows():
                tuple_data = tuple(
                    row[col] if col in row and pd.notna(row[col]) else None 
                    for col in columns
                )
                data_tuples.append(tuple_data)
            
            # Create INSERT query with ON CONFLICT handling
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            insert_query = f"""
            INSERT INTO {self.bronze_table} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT (url) DO UPDATE SET
                title = EXCLUDED.title,
                pub_date = EXCLUDED.pub_date,
                source = EXCLUDED.source,
                region = EXCLUDED.region,
                author = EXCLUDED.author,
                content = EXCLUDED.content,
                full_content = EXCLUDED.full_content,
                cleaned_at = EXCLUDED.cleaned_at,
                quality_check_status = EXCLUDED.quality_check_status,
                language_detected = EXCLUDED.language_detected,
                content_type = EXCLUDED.content_type,
                quality_score = EXCLUDED.quality_score,
                updated_at = CURRENT_TIMESTAMP
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    execute_values(
                        cursor,
                        insert_query,
                        data_tuples,
                        template=None,
                        page_size=1000
                    )
                    conn.commit()
            
            logger.info(f"Successfully saved {len(data_tuples)} records to bronze table")
            
        except Exception as e:
            logger.error(f"Error saving data to bronze table: {e}")
            raise
    
    def fetch_bronze_data(self, limit: Optional[int] = None, 
                         quality_filter: Optional[str] = None) -> pd.DataFrame:
        """Fetch bronze data for silver layer processing"""
        try:
            query = f"""
            SELECT 
                id, title, url, pub_date, source, region, author,
                content, full_content, language_detected, content_type,
                quality_check_status, quality_score, created_at
            FROM {self.bronze_table}
            WHERE 1=1
            """
            
            if quality_filter:
                query += f" AND quality_check_status = '{quality_filter}'"
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn)
            
            logger.info(f"Fetched {len(df)} bronze records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching bronze data: {e}")
            raise
    
    def mark_as_processed(self, record_ids: List[int]) -> None:
        """Mark bronze records as processed"""
        if not record_ids:
            return
        
        try:
            placeholders = ', '.join(['%s'] * len(record_ids))
            query = f"""
            UPDATE {self.bronze_table}
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, record_ids)
                    conn.commit()
            
            logger.info(f"Marked {len(record_ids)} records as processed")
            
        except Exception as e:
            logger.error(f"Error marking records as processed: {e}")
            raise
    
    def get_bronze_stats(self) -> Dict[str, Any]:
        """Get statistics about bronze data"""
        try:
            stats_query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN quality_check_status = 'high' THEN 1 END) as high_quality,
                COUNT(CASE WHEN quality_check_status = 'medium' THEN 1 END) as medium_quality,
                COUNT(CASE WHEN quality_check_status = 'low' THEN 1 END) as low_quality,
                COUNT(DISTINCT source) as unique_sources,
                COUNT(DISTINCT region) as unique_regions,
                COUNT(DISTINCT language_detected) as unique_languages,
                AVG(quality_score) as avg_quality_score,
                MIN(pub_date) as earliest_date,
                MAX(pub_date) as latest_date
            FROM {self.bronze_table}
            """
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(stats_query)
                    result = cursor.fetchone()
                    return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting bronze stats: {e}")
            raise
    
    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """Cleanup old bronze records"""
        try:
            cleanup_query = f"""
            DELETE FROM {self.bronze_table}
            WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '{days_to_keep} days'
            AND quality_check_status = 'low'
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(cleanup_query)
                    deleted_count = cursor.rowcount
                    conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old low-quality records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            raise