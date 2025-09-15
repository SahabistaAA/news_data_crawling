    enrichment_version VARCHAR(20)
CREATE TABLE IF NOT EXISTS silver_data (
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
