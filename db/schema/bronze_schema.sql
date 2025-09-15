CREATE TABLE IF NOT EXISTS bronze_data (
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
