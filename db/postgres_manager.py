import psycopg2 as ps
import psycopg2.extras as pse
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class PostgresManager:
    def __init__(
        self,
        dbname: str = "postgres",
        user: str = "postgres",
        password: str = "postgres",
        host: str = "localhost",
        port: int = 5432,
    ):
        self.conn = None
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connect()

    def connect(self):
        try:
            self.conn = ps.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
            )
            logging.info("Connected to PostgreSQL successfully.")
        except Exception as e:
            logging.error(f"Failed to connect to PostgreSQL: {e}")
            self.conn = None

    def close(self):
        if self.conn:
            self.conn.close()
            logging.info("PostgreSQL connection closed.")

    def execute(self, query: str, params: Optional[tuple] = None) -> bool:
        """Execute a single command."""
        if not self.conn:
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params or ())
                self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            self.conn.rollback()
            return False

    def fetch(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Fetch data from a query."""
        if not self.conn:
            return []
        try:
            with self.conn.cursor(cursor_factory=pse.DictCursor) as cur:
                cur.execute(query, params or ())
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching query: {e}")
            return []

    # =====================================================
    # RAW LAYER
    # =====================================================
    def save_raw_articles(self, articles: List[Dict], table: str = "raw_data") -> int:
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id SERIAL PRIMARY KEY,
            title TEXT,
            url TEXT,
            pub_date TIMESTAMP,
            source TEXT,
            region TEXT,
            author TEXT,
            content TEXT,
            full_content TEXT,
            ingested_at TIMESTAMP
        );
        """
        insert_query = f"""
        INSERT INTO {table} (
            title, url, pub_date, source, region, author, content, full_content, ingested_at
        ) VALUES %s
        ON CONFLICT (url) DO NOTHING;
        """
        values = [
            (
                a.get("title"),
                a.get("url"),
                a.get("pub_date"),
                a.get("source"),
                a.get("region"),
                a.get("author"),
                a.get("content"),
                a.get("full_content"),
                a.get("ingested_at"),
            )
            for a in articles
        ]
        return self._bulk_insert(values, create_table_query, insert_query, table)

    # =====================================================
    # BRONZE LAYER
    # =====================================================
    def save_bronze_articles(self, articles: List[Dict], table: str = "bronze_data") -> int:
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table} (
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
            is_duplicate BOOLEAN,
            quality_check_status VARCHAR(50)
        );
        """
        insert_query = f"""
        INSERT INTO {table} (
            title, url, pub_date, source, region, author, content, full_content,
            ingested_at, cleaned_at, is_duplicate, quality_check_status
        ) VALUES %s
        ON CONFLICT (url) DO NOTHING;
        """
        values = [
            (
                a.get("title"),
                a.get("url"),
                a.get("pub_date"),
                a.get("source"),
                a.get("region"),
                a.get("author"),
                a.get("content"),
                a.get("full_content"),
                a.get("ingested_at"),
                a.get("cleaned_at"),
                a.get("is_duplicate"),
                a.get("quality_check_status"),
            )
            for a in articles
        ]
        return self._bulk_insert(values, create_table_query, insert_query, table)

    # =====================================================
    # SILVER LAYER
    # =====================================================
    def save_silver_articles(self, articles: List[Dict], table: str = "silver_data") -> int:
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id SERIAL PRIMARY KEY,
            bronze_id INTEGER REFERENCES bronze_data(id),
            title TEXT,
            url TEXT UNIQUE,
            pub_date TIMESTAMP,
            source TEXT,
            region TEXT,
            content TEXT,
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
            enrichment_version VARCHAR(20)
        );
        """
        insert_query = f"""
        INSERT INTO {table} (
            bronze_id, title, url, pub_date, source, region, content,
            sentiment_score, sentiment_label, sentiment_confidence,
            emotions_score, entities, person_entities, organization_entities,
            location_entities, misc_entities, language_detected,
            processing_timestamp, enrichment_version
        ) VALUES %s
        ON CONFLICT (url) DO NOTHING;
        """
        values = [
            (
                a.get("bronze_id"),
                a.get("title"),
                a.get("url"),
                a.get("pub_date"),
                a.get("source"),
                a.get("region"),
                a.get("content"),
                a.get("sentiment_score"),
                a.get("sentiment_label"),
                a.get("sentiment_confidence"),
                a.get("emotions_score"),
                a.get("entities"),
                a.get("person_entities"),
                a.get("organization_entities"),
                a.get("location_entities"),
                a.get("misc_entities"),
                a.get("language_detected"),
                a.get("processing_timestamp"),
                a.get("enrichment_version"),
            )
            for a in articles
        ]
        return self._bulk_insert(values, create_table_query, insert_query, table)

    # =====================================================
    # GOLD LAYER
    # =====================================================
    def save_gold_articles(self, articles: List[Dict], table: str = "gold_data") -> int:
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id SERIAL PRIMARY KEY,
            silver_id INTEGER REFERENCES silver_data(id),
            pub_date DATE,
            source VARCHAR(255),
            region VARCHAR(100),
            author VARCHAR(255),
            topic_name VARCHAR(100),
            topic_id INTEGER,
            media_name VARCHAR(100),
            sentiment_label VARCHAR(20),
            avg_sentiment_score FLOAT,
            article_count INTEGER,
            sentiment_confidence FLOAT,
            key_entities JSONB,
            top_persons TEXT[],
            top_organizations TEXT[],
            top_locations TEXT[],
            person_entity_count INTEGER,
            organization_entity_count INTEGER,
            location_entity_count INTEGER,
            processing_date DATE
        );
        """
        insert_query = f"""
        INSERT INTO {table} (
            silver_id, pub_date, source, region, author, topic_name, topic_id, media_name,
            sentiment_label, avg_sentiment_score, article_count, sentiment_confidence,
            key_entities, top_persons, top_organizations, top_locations,
            person_entity_count, organization_entity_count, location_entity_count,
            processing_date
        ) VALUES %s
        ON CONFLICT DO NOTHING;
        """
        values = [
            (
                a.get("silver_id"),
                a.get("pub_date"),
                a.get("source"),
                a.get("region"),
                a.get("author"),
                a.get("topic_name"),
                a.get("topic_id"),
                a.get("media_name"),
                a.get("sentiment_label"),
                a.get("avg_sentiment_score"),
                a.get("article_count"),
                a.get("sentiment_confidence"),
                a.get("key_entities"),
                a.get("top_persons"),
                a.get("top_organizations"),
                a.get("top_locations"),
                a.get("person_entity_count"),
                a.get("organization_entity_count"),
                a.get("location_entity_count"),
                a.get("processing_date"),
            )
            for a in articles
        ]
        return self._bulk_insert(values, create_table_query, insert_query, table)

    # =====================================================
    # HELPER
    # =====================================================
    def _bulk_insert(self, values: List[tuple], create_table_query: str, insert_query: str, table: str) -> int:
        """Shared helper for bulk inserts"""
        if not self.conn:
            logger.error("PostgreSQL connection not initialized")
            return 0

        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_query)
                if values:
                    pse.execute_values(cur, insert_query, values)
                    self.conn.commit()
                    logger.info(f"Inserted {len(values)} records into {table}")
                    return len(values)
            return 0
        except Exception as e:
            logger.error(f"Error inserting into {table}: {e}")
            self.conn.rollback()
            return 0
