import os
import json
import logging
import time
from confluent_kafka import Producer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
from datetime import datetime

from crawlers.news_crawler import NewsCrawler

logger = logging.getLogger(__name__)

class NewsProducer:
    """Kafka producer for news articles."""

    def __init__(self, kafka_broker=None, topic_name=None, raw_path=None):
        self.kafka_broker = kafka_broker or os.getenv("KAFKA_BROKER", "kafka:9092")
        self.topic_name = topic_name or os.getenv("TOPIC_NAME", "news_topic")
        self.raw_path = raw_path or os.getenv("RAW_PATH", "data/raw/news_raw.json")

        # Instance counters for tracking
        self.successful_deliveries = 0
        self.failed_deliveries = 0

    def delivery_report(self, err, msg):
        """Handle delivery reports from Kafka producer."""
        if err:
            self.failed_deliveries += 1
            if err.code() == KafkaError._MSG_TIMED_OUT:
                logger.warning(f"Message timed out for topic {msg.topic() if msg else 'unknown'}")
            else:
                logger.error(f"Message delivery failed: {err}")
        else:
            self.successful_deliveries += 1
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] offset {msg.offset()}")

    def create_topic_if_not_exists(self, topic_name, broker):
        """Ensure topic exists before producing"""
        try:
            admin_client = AdminClient({
                'bootstrap.servers': broker,
                'client.id': 'topic-creator'
            })

            # Check if topic exists
            metadata = admin_client.list_topics(timeout=10)
            if topic_name not in metadata.topics:
                logger.info(f"Creating topic '{topic_name}'...")

                new_topics = [NewTopic(
                    topic_name,
                    num_partitions=3,
                    replication_factor=1
                )]

                fs = admin_client.create_topics(new_topics)
                for topic, f in fs.items():
                    try:
                        f.result(timeout=10)
                        logger.info(f"Topic '{topic}' created successfully")
                    except Exception as e:
                        logger.error(f"Failed to create topic: {e}")
                        return False

            return True
        except Exception as e:
            logger.error(f"Error managing topic: {e}")
            return False

    def wait_for_kafka(self, broker, max_retries=10, retry_delay=10):
        """Wait for Kafka to be available with more aggressive retrying"""
        logger.info(f"Testing Kafka connectivity to {broker}...")

        for attempt in range(max_retries):
            try:
                # Test multiple broker address formats
                test_brokers = [
                    broker,
                    "kafka:29092",
                    "kafka:9092",
                    "localhost:9092"
                ]

                for test_broker in test_brokers:
                    try:
                        logger.info(f"   Trying broker address: {test_broker}")
                        admin_client = AdminClient({
                            'bootstrap.servers': test_broker,
                            'client.id': 'kafka-health-check',
                            'socket.timeout.ms': 10000,
                            'api.version.request.timeout.ms': 10000,
                            'metadata.request.timeout.ms': 10000
                        })

                        # Try to list topics as a health check
                        metadata = admin_client.list_topics(timeout=15)
                        logger.info(f"Kafka is ready at {test_broker}. Found {len(metadata.topics)} topics.")

                        # Update the instance broker setting if this one works
                        self.kafka_broker = test_broker
                        return True

                    except Exception as broker_err:
                        logger.debug(f"   Failed with {test_broker}: {broker_err}")
                        continue

                logger.warning(f"All broker addresses failed (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

            except Exception as e:
                logger.warning(f"Kafka connectivity test failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        logger.error("Kafka is not available after multiple attempts and broker addresses")
        return False


    def run_producer(self):
        """
        Crawl news → save raw → publish to Kafka with robust error handling.
        """
        self.successful_deliveries = 0
        self.failed_deliveries = 0

        logger.info("Waiting for Kafka to be ready...")
        if not self.wait_for_kafka(self.kafka_broker):
            logger.error("Cannot proceed without Kafka connection")
            return 0

        logger.info("Ensuring topic exists...")
        if not self.create_topic_if_not_exists(self.topic_name, self.kafka_broker):
            logger.warning("Topic creation failed, but continuing...")

        logger.info("Crawling news sources...")
        # Import here to avoid circular imports if needed
        try:
            crawler = NewsCrawler(es_manager=None)
            result = crawler.crawl_all_sources()
            articles = [a.to_dict() for a in result["articles"]]
            # Reset article checker after crawl to avoid stale state if reused
            crawler.article_checker.reset()
        except Exception as e:
            logger.error(f"Failed to crawl articles: {e}")
            return 0

        if not articles:
            logger.warning("No articles to process")
            return 0

        logger.info(f"Preparing to send {len(articles)} articles to Kafka...")

        # Enhanced producer configuration (Python confluent-kafka compatible)
        producer_conf = {
            "bootstrap.servers": self.kafka_broker,
            "client.id": "news_producer_v2",

            # Timeout settings - more generous
            "message.timeout.ms": 30000,  # 30 seconds total timeout
            "request.timeout.ms": 10000,  # 10 seconds per request

            # Retry settings
            "retries": 5,
            "retry.backoff.ms": 1000,  # 1 second between retries

            # Batch settings for better throughput
            "batch.size": 16384,  # 16KB
            "linger.ms": 100,  # Wait up to 100ms to batch messages

            # Reliability settings
            "acks": "1",  # Wait for leader acknowledgment
            "enable.idempotence": False,  # Disable for simplicity

            # Compression
            "compression.type": "snappy",

            # Socket settings
            "socket.timeout.ms": 60000,  # 60 seconds
            "socket.keepalive.enable": True,

            # Connection settings
            "connections.max.idle.ms": 540000,  # 9 minutes
            "reconnect.backoff.ms": 50,
            "reconnect.backoff.max.ms": 1000,

            # Buffer settings (correct property names for confluent-kafka)
            "queue.buffering.max.messages": 100000,
            "queue.buffering.max.kbytes": 1048576,  # 1GB in KB
            "queue.buffering.max.ms": 1000,
        }

        producer = Producer(producer_conf)

        # Track processing
        sent_count = 0
        batch_size = 50  # Process in smaller batches

        try:
            for i, article in enumerate(articles, 1):
                try:
                    # Ensure all data is JSON serializable
                    clean_article = {}
                    for key, value in article.items():
                        if isinstance(value, datetime):
                            clean_article[key] = value.isoformat()
                        elif hasattr(value, 'isoformat'):
                            clean_article[key] = value.isoformat()
                        else:
                            clean_article[key] = value

                    # Convert article to JSON
                    message = json.dumps(clean_article, ensure_ascii=False)

                    # Produce to Kafka
                    producer.produce(
                        self.topic_name,
                        value=message.encode('utf-8'),
                        callback=self.delivery_report
                    )

                    sent_count += 1

                    # Poll periodically to trigger delivery reports
                    if i % 10 == 0:
                        producer.poll(0)
                        logger.info(f"Sent {sent_count}/{len(articles)} articles. Success: {self.successful_deliveries}, Failed: {self.failed_deliveries}")

                    # Flush periodically in smaller batches
                    if i % batch_size == 0:
                        logger.info(f"Flushing batch after {i} articles...")
                        producer.flush(timeout=10)
                        time.sleep(0.1)  # Small delay between batches

                except Exception as e:
                    logger.error(f"Failed to produce article {article.get('id', 'unknown')}: {e}")
                    continue

            # Final flush for remaining messages
            logger.info("Final flush of remaining messages...")
            producer.flush(timeout=30)

            # Final poll to get all delivery reports
            producer.poll(timeout=5)

        except KeyboardInterrupt:
            logger.warning("Producer interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error during production: {e}")
        finally:
            # Ensure producer is properly closed
            producer.flush(timeout=10)

        # Final status
        total_sent = self.successful_deliveries + self.failed_deliveries
        success_rate = (self.successful_deliveries / max(total_sent, 1)) * 100

        logger.info(f"Production Summary:")
        logger.info(f" - Articles sent: {sent_count}")
        logger.info(f" - Successful deliveries: {self.successful_deliveries}")
        logger.info(f" - Failed deliveries: {self.failed_deliveries}")
        logger.info(f" - Success rate: {success_rate:.1f}%")

        if self.failed_deliveries > 0:
            logger.warning(f"{self.failed_deliveries} messages failed delivery - check Kafka broker health")

        # Save to local file regardless of Kafka success
        self.save_to_local_file(articles)

        return self.successful_deliveries

    def run_producer_with_articles(self, articles):
        """
        Produce a list of articles to Kafka with robust error handling.
        """
        self.successful_deliveries = 0
        self.failed_deliveries = 0

        logger.info("Waiting for Kafka to be ready...")
        if not self.wait_for_kafka(self.kafka_broker):
            logger.error("Cannot proceed without Kafka connection")
            return 0

        logger.info("Ensuring topic exists...")
        if not self.create_topic_if_not_exists(self.topic_name, self.kafka_broker):
            logger.warning("Topic creation failed, but continuing...")

        if not articles:
            logger.warning("No articles to process")
            return 0

        logger.info(f"Preparing to send {len(articles)} articles to Kafka...")

        # Enhanced producer configuration (Python confluent-kafka compatible)
        producer_conf = {
            "bootstrap.servers": self.kafka_broker,
            "client.id": "news_producer_v2",

            # Timeout settings - more generous
            "message.timeout.ms": 30000,  # 30 seconds total timeout
            "request.timeout.ms": 10000,  # 10 seconds per request

            # Retry settings
            "retries": 5,
            "retry.backoff.ms": 1000,  # 1 second between retries

            # Batch settings for better throughput
            "batch.size": 16384,  # 16KB
            "linger.ms": 100,  # Wait up to 100ms to batch messages

            # Reliability settings
            "acks": "1",  # Wait for leader acknowledgment
            "enable.idempotence": False,  # Disable for simplicity

            # Compression
            "compression.type": "snappy",

            # Socket settings
            "socket.timeout.ms": 60000,  # 60 seconds
            "socket.keepalive.enable": True,

            # Connection settings
            "connections.max.idle.ms": 540000,  # 9 minutes
            "reconnect.backoff.ms": 50,
            "reconnect.backoff.max.ms": 1000,

            # Buffer settings (correct property names for confluent-kafka)
            "queue.buffering.max.messages": 100000,
            "queue.buffering.max.kbytes": 1048576,  # 1GB in KB
            "queue.buffering.max.ms": 1000,
        }

        producer = Producer(producer_conf)

        # Track processing
        sent_count = 0
        batch_size = 50  # Process in smaller batches

        try:
            for i, article in enumerate(articles, 1):
                try:
                    # Ensure all data is JSON serializable
                    clean_article = {}
                    for key, value in article.items():
                        if isinstance(value, datetime):
                            clean_article[key] = value.isoformat()
                        elif hasattr(value, 'isoformat'):
                            clean_article[key] = value.isoformat()
                        else:
                            clean_article[key] = value

                    # Convert article to JSON
                    message = json.dumps(clean_article, ensure_ascii=False)

                    # Produce to Kafka
                    producer.produce(
                        self.topic_name,
                        value=message.encode('utf-8'),
                        callback=self.delivery_report
                    )

                    sent_count += 1

                    # Poll periodically to trigger delivery reports
                    if i % 10 == 0:
                        producer.poll(0)
                        logger.info(f"Sent {sent_count}/{len(articles)} articles. Success: {self.successful_deliveries}, Failed: {self.failed_deliveries}")

                    # Flush periodically in smaller batches
                    if i % batch_size == 0:
                        logger.info(f"Flushing batch after {i} articles...")
                        producer.flush(timeout=10)
                        time.sleep(0.1)  # Small delay between batches

                except Exception as e:
                    logger.error(f"Failed to produce article {article.get('id', 'unknown')}: {e}")
                    continue

            # Final flush for remaining messages
            logger.info("Final flush of remaining messages...")
            producer.flush(timeout=30)

            # Final poll to get all delivery reports
            producer.poll(timeout=5)

        except KeyboardInterrupt:
            logger.warning("Producer interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error during production: {e}")
        finally:
            # Ensure producer is properly closed
            producer.flush(timeout=10)

        # Final status
        total_sent = self.successful_deliveries + self.failed_deliveries
        success_rate = (self.successful_deliveries / max(total_sent, 1)) * 100

        logger.info(f"Production Summary:")
        logger.info(f" - Articles sent: {sent_count}")
        logger.info(f" - Successful deliveries: {self.successful_deliveries}")
        logger.info(f" - Failed deliveries: {self.failed_deliveries}")
        logger.info(f" - Success rate: {success_rate:.1f}%")

        if self.failed_deliveries > 0:
            logger.warning(f"{self.failed_deliveries} messages failed delivery - check Kafka broker health")

        return self.successful_deliveries

    def save_to_local_file(self, articles):
        """Save articles to local file for backup"""
        from pathlib import Path

        try:
            file_path = Path(self.raw_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(articles)} raw articles to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save articles to file: {e}")


if __name__ == "__main__":
    # Set up logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('logs/news_producer.log', mode='a')  # File output
        ]
    )

    # Create producer instance and run
    producer = NewsProducer()
    result = producer.run_producer()
