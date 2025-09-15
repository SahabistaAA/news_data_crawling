#!/bin/bash
set -e

echo "Initializing Postgres..."

# Default credentials
POSTGRES_USER=${POSTGRES_USER:-airflow}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-airflow_11}
POSTGRES_DB=${POSTGRES_DB:-airflow}

# Wait for Postgres to be ready
until pg_isready -h localhost -p 5432 -U "$POSTGRES_USER"; do
  echo "Waiting for Postgres..."
  sleep 2
done

# Run schema initialization
psql -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<EOF
CREATE TABLE IF NOT EXISTS raw_articles (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    url TEXT UNIQUE,
    source TEXT,
    published_at TIMESTAMP,
    language VARCHAR(10),
    region VARCHAR(50),
    country VARCHAR(50),
    tags TEXT[]
);
EOF

echo "Postgres initialized successfully."
