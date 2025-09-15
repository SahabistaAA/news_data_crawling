import psycopg2, os, logging
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

class PostgresOperator:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "news"),
            user=os.getenv("POSTGRES_USER", "airflow"),
            password=os.getenv("POSTGRES_PASSWORD", "airflow_11"),
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=5432,
        )

    def save_raw_articles(self, articles):
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO articles (id, source_type, source_name, title, content, url, timestamp, metadata, full_content)
                VALUES %s
                ON CONFLICT (id) DO NOTHING;
                """,
                [(a["id"], a["source_type"], a["source_name"], a["title"],
                    a["content"], a["url"], a["timestamp"], json.dumps(a["metadata"]), a["full_content"])
                    for a in articles]
            )
        self.conn.commit()
