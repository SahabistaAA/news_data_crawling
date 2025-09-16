import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Ensure package imports resolve when running as a script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Crawling & Kafka
from crawlers.news_crawler import NewsCrawler
from producers.news_producers import NewsProducer
from consumers.news_consumers import NewsConsumer

# DB Managers
from db.postgres_manager import PostgresManager
from medallion_architect.bronze_layer.bronze_db import BronzeDB
from medallion_architect.silver_layer.silver_db import SilverDB

# Bronze steps
from medallion_architect.bronze_layer.cleaning import DataCleaner
from medallion_architect.bronze_layer.normalization import DataNormalizer
from medallion_architect.bronze_layer.standardization import DataStandardizer

# Silver enrichment
from medallion_architect.silver_layer.sentiment_analysis import SentimentAnalyzer
from medallion_architect.silver_layer.ner import EntityExtractor
from medallion_architect.silver_layer.topic_classification import TopicClassifier, KeywordExtractor
from medallion_architect.silver_layer.knowledge_graph import KnowledgeGraphBuilder


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("orchestrator")


def map_crawled_to_raw(article: Dict[str, Any]) -> Dict[str, Any]:
    """Map CrawledData dict into raw_data schema fields for PostgreSQL."""
    metadata = article.get("metadata", {}) or {}
    pub_date = metadata.get("pub_date")
    try:
        # pass through; DB layer converts as needed
        pub_date_val = pub_date
    except Exception:
        pub_date_val = None

    return {
        "title": article.get("title"),
        "url": article.get("url"),
        "pub_date": pub_date_val,
        "source": article.get("source_name"),
        "region": metadata.get("region"),
        "author": metadata.get("author", None),
        "content": article.get("content"),
        "full_content": article.get("full_content"),
        "ingested_at": datetime.utcnow(),
    }


def step_crawl() -> List[Dict[str, Any]]:
    logger.info("Crawling news sources...")
    crawler = NewsCrawler()
    result = crawler.crawl_all_sources()
    articles = [a.to_dict() for a in result.get("articles", [])]
    logger.info(f"Crawled {len(articles)} articles across {result.get('feed_count', 0)} feeds")
    return articles


def step_save_raw_to_postgres(articles: List[Dict[str, Any]]) -> int:
    logger.info("Saving raw articles to PostgreSQL...")
    if not articles:
        logger.info("No articles to save to raw_data")
        return 0

    mgr = PostgresManager()
    mapped = [map_crawled_to_raw(a) for a in articles]
    try:
        n = mgr.save_raw_articles(mapped, table="raw_data")
        logger.info(f"Saved {n} raw records to PostgreSQL")
        return n
    except Exception as e:
        logger.error(f"Failed to save raw articles: {e}")
        return 0


def step_produce_kafka(articles: List[Dict[str, Any]]) -> int:
    logger.info("Producing articles to Kafka...")
    if not articles:
        logger.info("No articles to produce to Kafka")
        return 0
    producer = NewsProducer({"bootstrap.servers": "localhost:9092"})
    # Articles must be JSON-serializable dicts; map datetime fields
    serializable = []
    for a in articles:
        clean = {}
        for k, v in a.items():
            if isinstance(v, datetime):
                clean[k] = v.isoformat()
            else:
                clean[k] = v
        serializable.append(clean)
    try:
        sent = producer.run_producer_with_articles(serializable)
        logger.info(f"Produced {sent} messages to Kafka")
        return sent
    except Exception as e:
        logger.error(f"Kafka production failed: {e}")
        return 0


def step_consume_kafka(limit: int = 100) -> List[Dict[str, Any]]:
    logger.info("Consuming messages from Kafka and indexing to Elasticsearch...")
    consumer = NewsConsumer()
    try:
        docs = consumer.run_consumer(limit=limit)
        logger.info(f"Consumed and indexed {len(docs)} documents to Elasticsearch")
        return docs
    except Exception as e:
        logger.error(f"Kafka consumption failed: {e}")
        return []


def step_bronze_process(batch_limit: int = None) -> int:
    logger.info("Running Bronze layer processing (clean, normalize, standardize)...")
    bronze_db = BronzeDB()
    cleaner = DataCleaner()
    normalizer = DataNormalizer()
    standardizer = DataStandardizer()

    # Ensure table exists
    bronze_db.create_bronze_table()

    # Fetch raw data
    raw_df = bronze_db.fetch_raw_data(limit=batch_limit)
    if raw_df is None or raw_df.empty:
        logger.info("No raw data to process for bronze")
        return 0

    # Process bronze pipeline
    cleaned = cleaner.clean_data(raw_df)
    normalized = normalizer.normalize_data(cleaned)
    standardized = standardizer.standardize_data(normalized)

    # Save to bronze table
    bronze_db.save_to_bronze(standardized)
    logger.info(f"Bronze processing completed: {len(standardized)} records saved")
    return len(standardized)


def _silver_enrich_batch(df, sentiment_analyzer, entity_extractor, topic_classifier, keyword_extractor):
    import pandas as pd
    enriched = df.copy()

    # Sentiment
    sentiments = []
    for _, row in enriched.iterrows():
        content = str(row.get('content', ''))
        language = str(row.get('language_detected', 'english'))
        try:
            sentiments.append(sentiment_analyzer.analyze_sentiment(content, language))
        except Exception:
            sentiments.append({'score': 0.0, 'label': 'neutral', 'confidence': 0.0, 'emotion_scores': {}})

    enriched['sentiment_score'] = [s.get('score', 0.0) for s in sentiments]
    enriched['sentiment_label'] = [s.get('label', 'neutral') for s in sentiments]
    enriched['sentiment_confidence'] = [s.get('confidence', 0.0) for s in sentiments]
    enriched['emotions_score'] = [s.get('emotion_scores', {}) for s in sentiments]

    # NER
    entities = []
    persons = []
    orgs = []
    locs = []
    miscs = []
    for _, row in enriched.iterrows():
        text = f"{str(row.get('title',''))}. {str(row.get('content',''))}"
        language = str(row.get('language_detected', 'english'))
        try:
            ent_res = entity_extractor.extract_entities(text, language)
        except Exception:
            ent_res = {'entities': [], 'person_entities': [], 'organization_entities': [], 'location_entities': [], 'misc_entities': []}
        entities.append(ent_res.get('entities', []))
        persons.append(ent_res.get('person_entities', []))
        orgs.append(ent_res.get('organization_entities', []))
        locs.append(ent_res.get('location_entities', []))
        miscs.append(ent_res.get('misc_entities', []))

    enriched['entities'] = entities
    enriched['person_entities'] = persons
    enriched['organization_entities'] = orgs
    enriched['location_entities'] = locs
    enriched['misc_entities'] = miscs

    # Topics and keywords
    topics = []
    keywords = []
    for _, row in enriched.iterrows():
        try:
            topics.append(topic_classifier.classify_topic(str(row.get('content','')), str(row.get('title',''))))
        except Exception:
            topics.append({'primary': 'General', 'secondary': [], 'confidence': 0.0})
        try:
            keywords.append(keyword_extractor.extract_keywords(str(row.get('content','')), str(row.get('title','')), 10))
        except Exception:
            keywords.append([])

    enriched['topic_classification'] = topics
    enriched['keywords'] = keywords

    # Metrics
    enriched['person_entity_count'] = [len(x) for x in persons]
    enriched['organization_entity_count'] = [len(x) for x in orgs]
    enriched['location_entity_count'] = [len(x) for x in locs]
    enriched['total_entity_count'] = enriched['person_entity_count'] + enriched['organization_entity_count'] + enriched['location_entity_count']

    def _content_richness(row):
        score = 0.0
        types = 0
        if len(row.get('person_entities', [])) > 0:
            types += 1
        if len(row.get('organization_entities', [])) > 0:
            types += 1
        if len(row.get('location_entities', [])) > 0:
            types += 1
        score += min(types / 3.0, 1.0) * 0.4
        score += float(row.get('sentiment_confidence', 0.0)) * 0.2
        clen = len(str(row.get('content', '')))
        if clen > 1000:
            lscore = 1.0
        elif clen > 500:
            lscore = 0.8
        elif clen > 200:
            lscore = 0.6
        else:
            lscore = 0.3
        score += lscore * 0.2
        if str(row.get('author','')):
            score += 0.1
        score += float(row.get('quality_score', 0.0)) * 0.05
        tc = row.get('topic_classification', {}) or {}
        if isinstance(tc, dict):
            score += float(tc.get('confidence', 0.0)) * 0.05
        kw = row.get('keywords', []) or []
        if kw:
            score += min(len(kw)/10.0, 1.0) * 0.05
        return round(min(score, 1.0), 3)

    def _info_density(row):
        words = len(str(row.get('content','')).split())
        ents = int(row.get('total_entity_count', 0))
        if words <= 0:
            return 0.0
        density = (ents / words) * 100.0
        return round(min(density/5.0, 1.0), 3)

    enriched['content_richness_score'] = enriched.apply(_content_richness, axis=1)
    enriched['information_density'] = enriched.apply(_info_density, axis=1)

    # Add required fields for SilverDB
    enriched['processing_timestamp'] = datetime.utcnow()
    enriched['enrichment_version'] = 'v1.0.0'
    if 'content_type' not in enriched.columns:
        enriched['content_type'] = 'news'

    return enriched


def step_silver_process(batch_size: int = 100) -> int:
    logger.info("Running Silver layer enrichment (NER, sentiment, topics, keywords) and Neo4j KG...")

    silver_db = SilverDB()
    silver_db.create_silver_table()

    # Initialize models with fallbacks
    sentiment_analyzer = SentimentAnalyzer()
    entity_extractor = EntityExtractor()
    topic_classifier = TopicClassifier()
    keyword_extractor = KeywordExtractor()

    try:
        sentiment_analyzer.load_models()
    except Exception as e:
        logger.warning(f"Sentiment models could not be fully loaded: {e}")
    try:
        entity_extractor.load_models()
    except Exception as e:
        logger.warning(f"NER models could not be fully loaded: {e}")

    # Fetch bronze data for processing
    import pandas as pd
    bronze_batch = silver_db.fetch_bronze_for_processing(batch_size=batch_size)
    if bronze_batch is None or bronze_batch.empty:
        logger.info("No bronze data to process for silver")
        return 0

    enriched = _silver_enrich_batch(
        bronze_batch,
        sentiment_analyzer,
        entity_extractor,
        topic_classifier,
        keyword_extractor,
    )

    # Save enriched data to silver table
    silver_db.save_to_silver(enriched)

    # Build knowledge graph in Neo4j
    try:
        kg = KnowledgeGraphBuilder()
        kg.create_knowledge_graph(enriched)
        kg.close()
    except Exception as e:
        logger.warning(f"Knowledge graph step skipped/failed: {e}")

    logger.info(f"Silver processing completed: {len(enriched)} records saved & graphed")
    return len(enriched)


def main():
    parser = argparse.ArgumentParser(description="Run News Data Pipeline (crawl -> kafka -> db -> bronze -> silver -> KG)")
    parser.add_argument('--crawl', action='store_true', help='Run crawling step')
    parser.add_argument('--save-raw', action='store_true', help='Save crawled raw data to PostgreSQL')
    parser.add_argument('--produce', action='store_true', help='Produce crawled data to Kafka')
    parser.add_argument('--consume', action='store_true', help='Consume from Kafka and index to Elasticsearch')
    parser.add_argument('--bronze', action='store_true', help='Process Bronze layer (raw -> bronze)')
    parser.add_argument('--silver', action='store_true', help='Process Silver layer (bronze -> silver + KG)')
    parser.add_argument('--limit', type=int, default=100, help='Batch limit for consume/silver steps')

    args = parser.parse_args()

    # If no flags are set, run the full pipeline
    run_all = not any([args.crawl, args.save_raw, args.produce, args.consume, args.bronze, args.silver])

    articles: List[Dict[str, Any]] = []

    if args.crawl or run_all:
        articles = step_crawl()

    if args.save_raw or run_all:
        if not articles:
            # If not crawled in this run, try to crawl quickly
            if not articles:
                articles = step_crawl()
        step_save_raw_to_postgres(articles)

    if args.produce or run_all:
        if not articles:
            if not articles:
                articles = step_crawl()
        step_produce_kafka(articles)

    if args.consume or run_all:
        step_consume_kafka(limit=args.limit)

    if args.bronze or run_all:
        step_bronze_process()

    if args.silver or run_all:
        step_silver_process(batch_size=args.limit)


if __name__ == '__main__':
    main()
