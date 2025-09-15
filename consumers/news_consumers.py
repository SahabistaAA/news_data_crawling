import os
import sys
import json
import logging
from datetime import datetime
from confluent_kafka import Consumer
from dateutil import parser

from db.elasticsearch_manager import ElasticsearchManager

logger = logging.getLogger(__name__)

class NewsConsumer:
    """Kafka consumer for news articles."""

    def __init__(self, kafka_broker=None, topic_name=None, group_id=None, es_host=None, index_name="news"):
        self.kafka_broker = kafka_broker or os.getenv("KAFKA_BROKER", "localhost:9092")
        self.topic_name = topic_name or os.getenv("TOPIC_NAME", "news_topic")
        self.group_id = group_id or os.getenv("GROUP_ID", "news-consumer-group")
        self.es_host = es_host or os.getenv("ELASTICSEARCH_HOST", "localhost")
        self.index_name = index_name

        self.consumer_conf = {
            "bootstrap.servers": self.kafka_broker,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
        }
        self.consumer = Consumer(self.consumer_conf)

    def parse_and_format_date(self, date_str):
        """Parse various date formats and convert to ISO 8601 format that Elasticsearch accepts."""
        try:
            # Parse the date string
            parsed_date = parser.parse(date_str)
            # Convert to ISO 8601 format with timezone (Elasticsearch preferred format)
            return parsed_date.isoformat()
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse date '{date_str}': {e}")
            return None

    def normalize_article_dates(self, article):
        """Normalize all date fields in the article to ISO 8601 format."""
        # Handle metadata.pub_date
        metadata = article.get("metadata", {})
        pub_date_str = metadata.get("pub_date")
        if pub_date_str:
            normalized_date = self.parse_and_format_date(pub_date_str)
            if normalized_date:
                article["metadata"]["pub_date"] = normalized_date
            else:
                # If we can't parse it, remove it to avoid Elasticsearch errors
                del article["metadata"]["pub_date"]
                logger.warning(f"Removed invalid pub_date: {pub_date_str}")
        
        return article

    def run_consumer(self, topic=None, limit=None):
        topic = topic or self.topic_name
        es_host = self.es_host.replace('http://', '').replace('https://', '')

        # Initialize Elasticsearch manager
        try:
            es_manager = ElasticsearchManager(host=es_host, port=9200)
            logger.info(f"Connected to Elasticsearch at {es_host}:9200")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return []

        self.consumer.subscribe([topic])
        logger.info(f"Subscribed to Kafka topic '{topic}' with broker '{self.kafka_broker}'")

        all_articles = []
        count = 0
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    logger.debug("No message received, continuing...")
                    continue
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue

                try:
                    article = json.loads(msg.value().decode("utf-8"))
                    article["processed_at"] = datetime.now().isoformat()

                    # Normalize all date fields before indexing
                    article = self.normalize_article_dates(article)

                    all_articles.append(article)

                    # Index to Elasticsearch
                    es_manager.index_document(self.index_name, article)
                    logger.info(f"Indexed article {article.get('id', 'unknown')} to Elasticsearch")

                    count += 1
                    if limit and count >= limit:
                        logger.info(f"Reached message limit ({limit}), stopping consumer.")
                        break

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message JSON: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue

        except KeyboardInterrupt:
            logger.warning("Consumer interrupted, closing...")
        except Exception as e:
            logger.error(f"Unexpected error in consumer: {e}")
        finally:
            self.consumer.close()
            logger.info("Consumer closed")
            logger.info(f"Total articles processed: {len(all_articles)}")

        return all_articles


if __name__ == "__main__":
    # Set up logging to both console and file
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('logs/news_consumer.log', mode='a')  # File output
        ]
    )

    logger.info("Starting news consumer...")
    logger.info(f"Kafka Broker: {os.getenv('KAFKA_BROKER', 'localhost:9092')}")
    logger.info(f"Topic: {os.getenv('TOPIC_NAME', 'news_topic')}")
    logger.info(f"Elasticsearch Host: {os.getenv('ELASTICSEARCH_HOST', 'localhost')}")
    logger.info("All data will be indexed to Elasticsearch instead of saved to local files")

    consumer = NewsConsumer()
    consumer.run_consumer()
