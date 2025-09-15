import datetime
from postgres_manager import PostgresManager
from elasticsearch_manager import ElasticsearchManager
from neo4j_manager import Neo4jManager

# Sample article
sample_article = {
    "title": "AI is Transforming News Analytics",
    "content": "Artificial Intelligence is being used to enrich news pipelines...",
    "url": "https://example.com/ai-news",
    "source": "TechCrunch",
    "published_at": datetime.datetime.utcnow(),
    "language": "en",
    "region": "Asia",
    "country": "Indonesia",
    "tags": ["AI", "News", "Analytics"]
}

def main():
    # Postgres
    pg = PostgresManager()
    pg.connect()
    pg.save_raw_articles([sample_article])
    print("Inserted sample into Postgres")

    # Elasticsearch
    es = ElasticsearchManager()
    es.create_index("news")
    es.insert_document("news", sample_article["url"], sample_article)
    print("Inserted sample into Elasticsearch")

    # Neo4j
    neo = Neo4jManager()
    neo.create_news_node(sample_article)
    neo.create_relationship("News", "url", sample_article["url"], "Publisher", "name", sample_article["source"], "PUBLISHED_BY")
    print("Inserted sample into Neo4j")

if __name__ == "__main__":
    main()
