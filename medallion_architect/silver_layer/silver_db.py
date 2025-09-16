import psycopg2 as pg2
import pandas as pd
from psycopg2.extras import execute_values, RealDictCursor, Json
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class SilverDB:
    """Database operations for Silver layer"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'airflow'),
            'user': os.getenv('DB_USER', 'airflow'),
            'password': os.getenv('DB_PASSWORD', 'airflow_11')
        }
        
        self.silver_table = 'silver_data'
    
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
    
    def create_silver_table(self) -> None:
        """Create silver table if it doesn't exist"""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.silver_table} (
            id SERIAL PRIMARY KEY,
            bronze_id INTEGER,
            title TEXT,
            url TEXT UNIQUE,
            pub_date TIMESTAMP,
            source TEXT,
            region TEXT,
            content TEXT,
            topic_classification JSONB,
            keywords TEXT[],
            sentiment_score FLOAT,
            sentiment_label VARCHAR(20),
            sentiment_confidence FLOAT,
            emotions_score JSONB,
            entities JSONB,
            person_entities TEXT[],
            organization_entities TEXT[],
            location_entities TEXT[],
            misc_entities TEXT[],
            language_detected VARCHAR(10),
            processing_timestamp TIMESTAMP,
            enrichment_version VARCHAR(20),
            content_type VARCHAR(50),
            quality_score FLOAT,
            person_entity_count INTEGER DEFAULT 0,
            organization_entity_count INTEGER DEFAULT 0,
            location_entity_count INTEGER DEFAULT 0,
            total_entity_count INTEGER DEFAULT 0,
            content_richness_score FLOAT DEFAULT 0.0,
            information_density FLOAT DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_silver_url ON {self.silver_table}(url);
        CREATE INDEX IF NOT EXISTS idx_silver_source ON {self.silver_table}(source);
        CREATE INDEX IF NOT EXISTS idx_silver_region ON {self.silver_table}(region);
        CREATE INDEX IF NOT EXISTS idx_silver_pub_date ON {self.silver_table}(pub_date);
        CREATE INDEX IF NOT EXISTS idx_silver_sentiment ON {self.silver_table}(sentiment_label);
        CREATE INDEX IF NOT EXISTS idx_silver_language ON {self.silver_table}(language_detected);
        CREATE INDEX IF NOT EXISTS idx_silver_content_type ON {self.silver_table}(content_type);
        CREATE INDEX IF NOT EXISTS idx_silver_quality ON {self.silver_table}(quality_score);
        
        -- Create GIN indexes for JSONB columns
        CREATE INDEX IF NOT EXISTS idx_silver_emotions_gin ON {self.silver_table} USING GIN (emotions_score);
        CREATE INDEX IF NOT EXISTS idx_silver_entities_gin ON {self.silver_table} USING GIN (entities);
        
        -- Create array indexes
        CREATE INDEX IF NOT EXISTS idx_silver_persons_gin ON {self.silver_table} USING GIN (person_entities);
        CREATE INDEX IF NOT EXISTS idx_silver_orgs_gin ON {self.silver_table} USING GIN (organization_entities);
        CREATE INDEX IF NOT EXISTS idx_silver_locations_gin ON {self.silver_table} USING GIN (location_entities);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_query)
                    conn.commit()
            logger.info(f"Silver table '{self.silver_table}' created/verified successfully")
        except Exception as e:
            logger.error(f"Error creating silver table: {e}")
            raise

    def fetch_bronze_for_processing(self, batch_size: int = 10) -> pd.DataFrame:
        """Fetch bronze data for silver processing"""
        try:
            query = f"""
            SELECT * FROM bronze_data
            WHERE quality_check_status IN ('high', 'medium')
            ORDER BY id DESC
            LIMIT {batch_size}
            """

            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn)

            logger.info(f"Fetched {len(df)} bronze records for processing")
            return df

        except Exception as e:
            logger.error(f"Error fetching bronze data: {e}")
            raise

    def save_to_silver(self, df: pd.DataFrame) -> None:
        """Save enriched data to silver table"""
        if df.empty:
            logger.warning("No data to save to silver table")
            return
        
        try:
            # Prepare column list for insertion
            columns = [
                'bronze_id', 'title', 'url', 'pub_date', 'source', 'region',
                'content', 'topic_classification', 'keywords', 'sentiment_score', 'sentiment_label', 'sentiment_confidence',
                'emotions_score', 'entities', 'person_entities', 'organization_entities',
                'location_entities', 'misc_entities', 'language_detected',
                'processing_timestamp', 'enrichment_version', 'content_type',
                'quality_score', 'person_entity_count', 'organization_entity_count',
                'location_entity_count', 'total_entity_count', 'content_richness_score',
                'information_density'
            ]
            
            # Ensure all columns exist in DataFrame
            for col in columns:
                if col not in df.columns:
                    if col == 'processing_timestamp':
                        df[col] = pd.Timestamp.now()
                    elif col == 'enrichment_version':
                        df[col] = 'v1.0.0'
                    elif col in ['sentiment_score', 'sentiment_confidence', 'quality_score',
                               'content_richness_score', 'information_density']:
                        df[col] = 0.0
                    elif col in ['person_entity_count', 'organization_entity_count',
                               'location_entity_count', 'total_entity_count']:
                        df[col] = 0
                    elif col in ['person_entities', 'organization_entities',
                               'location_entities', 'misc_entities']:
                        df[col] = df[col].apply(lambda x: [] if pd.isna(x) else x)
                    elif col in ['emotions_score', 'entities']:
                        df[col] = df[col].apply(lambda x: {} if pd.isna(x) else x)
                    else:
                        df[col] = ''
            
            # Prepare data for insertion
            data_tuples = []
            for _, row in df.iterrows():
                # Handle JSONB fields
                emotions_score = row['emotions_score']
                if isinstance(emotions_score, str):
                    try:
                        emotions_score = json.loads(emotions_score)
                    except:
                        emotions_score = {}
                
                entities = row['entities']
                if isinstance(entities, str):
                    try:
                        entities = json.loads(entities)
                    except:
                        entities = []
                
                # Handle array fields
                person_entities = row['person_entities']
                if isinstance(person_entities, str):
                    try:
                        person_entities = json.loads(person_entities)
                    except:
                        person_entities = []
                
                organization_entities = row['organization_entities']
                if isinstance(organization_entities, str):
                    try:
                        organization_entities = json.loads(organization_entities)
                    except:
                        organization_entities = []
                
                location_entities = row['location_entities']
                if isinstance(location_entities, str):
                    try:
                        location_entities = json.loads(location_entities)
                    except:
                        location_entities = []
                
                misc_entities = row['misc_entities']
                if isinstance(misc_entities, str):
                    try:
                        misc_entities = json.loads(misc_entities)
                    except:
                        misc_entities = []
                
                # Handle topic_classification
                topic_classification = row.get('topic_classification', {})
                if isinstance(topic_classification, str):
                    try:
                        topic_classification = json.loads(topic_classification)
                    except:
                        topic_classification = {}

                # Handle keywords
                keywords = row.get('keywords', [])
                if isinstance(keywords, str):
                    try:
                        keywords = json.loads(keywords)
                    except:
                        keywords = []

                tuple_data = (
                    row.get('id'),  # bronze_id
                    row.get('title'),
                    row.get('url'),
                    row.get('pub_date'),
                    row.get('source'),
                    row.get('region'),
                    row.get('content'),
                    Json(topic_classification),
                    keywords,
                    float(row.get('sentiment_score', 0.0)),
                    row.get('sentiment_label', 'neutral'),
                    float(row.get('sentiment_confidence', 0.0)),
                    Json(emotions_score),
                    Json(entities),
                    person_entities,
                    organization_entities,
                    location_entities,
                    misc_entities,
                    row.get('language_detected', 'unknown'),
                    row.get('processing_timestamp'),
                    row.get('enrichment_version', 'v1.0.0'),
                    row.get('content_type', 'news'),
                    float(row.get('quality_score', 0.0)),
                    int(row.get('person_entity_count', 0)),
                    int(row.get('organization_entity_count', 0)),
                    int(row.get('location_entity_count', 0)),
                    int(row.get('total_entity_count', 0)),
                    float(row.get('content_richness_score', 0.0)),
                    float(row.get('information_density', 0.0))
                )
                data_tuples.append(tuple_data)
            
            # Create INSERT query with ON CONFLICT handling
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            insert_query = f"""
            INSERT INTO {self.silver_table} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT (url) DO UPDATE SET
                topic_classification = EXCLUDED.topic_classification,
                keywords = EXCLUDED.keywords,
                sentiment_score = EXCLUDED.sentiment_score,
                sentiment_label = EXCLUDED.sentiment_label,
                sentiment_confidence = EXCLUDED.sentiment_confidence,
                emotions_score = EXCLUDED.emotions_score,
                entities = EXCLUDED.entities,
                person_entities = EXCLUDED.person_entities,
                organization_entities = EXCLUDED.organization_entities,
                location_entities = EXCLUDED.location_entities,
                misc_entities = EXCLUDED.misc_entities,
                processing_timestamp = EXCLUDED.processing_timestamp,
                enrichment_version = EXCLUDED.enrichment_version,
                quality_score = EXCLUDED.quality_score,
                person_entity_count = EXCLUDED.person_entity_count,
                organization_entity_count = EXCLUDED.organization_entity_count,
                location_entity_count = EXCLUDED.location_entity_count,
                total_entity_count = EXCLUDED.total_entity_count,
                content_richness_score = EXCLUDED.content_richness_score,
                information_density = EXCLUDED.information_density,
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
            
            logger.info(f"Successfully saved {len(data_tuples)} records to silver table")
            
        except Exception as e:
            logger.error(f"Error saving data to silver table: {e}")
            raise
    
    def fetch_silver_data(self, limit: Optional[int] = None,
                         sentiment_filter: Optional[str] = None,
                         quality_threshold: Optional[float] = None) -> pd.DataFrame:
        """Fetch silver data for gold layer processing"""
        try:
            query = f"""
            SELECT
                id, bronze_id, title, url, pub_date, source, region, content,
                topic_classification, keywords, sentiment_score, sentiment_label, sentiment_confidence,
                emotions_score, entities, person_entities, organization_entities,
                location_entities, misc_entities, language_detected,
                content_type, quality_score, person_entity_count,
                organization_entity_count, location_entity_count,
                total_entity_count, content_richness_score,
                information_density, created_at
            FROM {self.silver_table}
            WHERE 1=1
            """
            
            params = []
            
            if sentiment_filter:
                query += " AND sentiment_label = %s"
                params.append(sentiment_filter)
            
            if quality_threshold:
                query += " AND quality_score >= %s"
                params.append(quality_threshold)
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            logger.info(f"Fetched {len(df)} silver records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching silver data: {e}")
            raise
    
    def get_silver_stats(self) -> Dict[str, Any]:
        """Get statistics about silver data"""
        try:
            stats_query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN sentiment_label = 'positive' THEN 1 END) as positive_sentiment,
                COUNT(CASE WHEN sentiment_label = 'negative' THEN 1 END) as negative_sentiment,
                COUNT(CASE WHEN sentiment_label = 'neutral' THEN 1 END) as neutral_sentiment,
                COUNT(DISTINCT source) as unique_sources,
                COUNT(DISTINCT region) as unique_regions,
                COUNT(DISTINCT language_detected) as unique_languages,
                AVG(sentiment_score) as avg_sentiment_score,
                AVG(quality_score) as avg_quality_score,
                AVG(content_richness_score) as avg_content_richness,
                AVG(information_density) as avg_information_density,
                AVG(total_entity_count) as avg_entity_count,
                MIN(pub_date) as earliest_date,
                MAX(pub_date) as latest_date
            FROM {self.silver_table}
            """
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(stats_query)
                    result = cursor.fetchone()
                    return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting silver stats: {e}")
            raise
    
    def get_entity_analysis(self) -> Dict[str, Any]:
        """Get entity analysis from silver data"""
        try:
            entity_query = f"""
            SELECT 
                unnest(person_entities) as entity,
                'person' as entity_type,
                COUNT(*) as frequency
            FROM {self.silver_table}
            WHERE array_length(person_entities, 1) > 0
            
            UNION ALL
            
            SELECT 
                unnest(organization_entities) as entity,
                'organization' as entity_type,
                COUNT(*) as frequency
            FROM {self.silver_table}
            WHERE array_length(organization_entities, 1) > 0
            
            UNION ALL
            
            SELECT 
                unnest(location_entities) as entity,
                'location' as entity_type,
                COUNT(*) as frequency
            FROM {self.silver_table}
            WHERE array_length(location_entities, 1) > 0
            """
            
            with self.get_connection() as conn:
                df = pd.read_sql_query(entity_query, conn)
            
            # Group by entity type and get top entities
            result = {}
            for entity_type in ['person', 'organization', 'location']:
                type_data = df[df['entity_type'] == entity_type]
                top_entities = type_data.nlargest(10, 'frequency')[['entity', 'frequency']].to_dict('records')
                result[f'top_{entity_type}_entities'] = top_entities
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting entity analysis: {e}")
            return {}
    
    def search_articles(self, search_params: Dict[str, Any]) -> pd.DataFrame:
        """Search articles based on various parameters"""
        try:
            query = f"""
            SELECT
                id, title, url, pub_date, source, region,
                topic_classification, keywords, sentiment_score, sentiment_label, quality_score,
                person_entities, organization_entities, location_entities
            FROM {self.silver_table}
            WHERE 1=1
            """
            
            params = []
            
            # Text search in title or content
            if search_params.get('text'):
                query += " AND (title ILIKE %s OR content ILIKE %s)"
                search_text = f"%{search_params['text']}%"
                params.extend([search_text, search_text])
            
            # Date range filter
            if search_params.get('start_date'):
                query += " AND pub_date >= %s"
                params.append(search_params['start_date'])
            
            if search_params.get('end_date'):
                query += " AND pub_date <= %s"
                params.append(search_params['end_date'])
            
            # Source filter
            if search_params.get('source'):
                query += " AND source = %s"
                params.append(search_params['source'])
            
            # Region filter
            if search_params.get('region'):
                query += " AND region = %s"
                params.append(search_params['region'])
            
            # Sentiment filter
            if search_params.get('sentiment'):
                query += " AND sentiment_label = %s"
                params.append(search_params['sentiment'])
            
            # Entity search
            if search_params.get('entity'):
                entity = search_params['entity']
                query += """ AND (
                    %s = ANY(person_entities) OR 
                    %s = ANY(organization_entities) OR 
                    %s = ANY(location_entities)
                )"""
                params.extend([entity, entity, entity])
            
            query += " ORDER BY pub_date DESC LIMIT 1000"
            
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return pd.DataFrame()
    
    def cleanup_old_records(self, days_to_keep: int = 90) -> int:
        """Cleanup old silver records"""
        try:
            cleanup_query = f"""
            DELETE FROM {self.silver_table}
            WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '{days_to_keep} days'
            AND quality_score < 0.3
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(cleanup_query)
                    deleted_count = cursor.rowcount
                    conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old low-quality silver records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            raise