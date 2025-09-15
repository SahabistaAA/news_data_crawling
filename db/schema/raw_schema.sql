CREATE TABLE IF NOT EXISTS raw_data (
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
