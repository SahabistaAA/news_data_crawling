import pytest
import unittest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from confluent_kafka import KafkaError
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
subparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# Import the classes to test
from news_data_crawling.consumers.news_consumers import NewsConsumer
from news_data_crawling.producers.news_producers import NewsProducer

class TestNewsConsumer:
    """Test NewsConsumer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.consumer = NewsConsumer(
            kafka_broker="localhost:9092",
            topic_name="test_topic",
            group_id="test_group",
            es_host="localhost"
        )
        
        # Sample test article
        self.sample_article = {
            "id": "test-article-123",
            "title": "Test Article",
            "content": "Test content",
            "metadata": {
                "pub_date": "2023-01-01T12:00:00Z",
                "region": "test",
                "country": "Testland"
            }
        }

    @patch('news_data_crawling.db.elasticsearch_manager.ElasticsearchManager')
    @patch('news_data_crawling.consumers.news_consumers.NewsConsumer')
    def test_consumer_initialization(self, mock_consumer, mock_es_manager):
        """Test consumer initialization"""
        assert self.consumer.kafka_broker == "localhost:9092"
        assert self.consumer.topic_name == "test_topic"
        assert self.consumer.group_id == "test_group"
        assert self.consumer.es_host == "localhost"
        mock_consumer.assert_called_once()

    def test_parse_and_format_date_valid(self):
        """Test date parsing with valid dates"""
        test_dates = [
            "2023-01-01T12:00:00Z",
            "January 1, 2023",
            "2023-01-01",
            "01/01/2023"
        ]
        
        for date_str in test_dates:
            result = self.consumer.parse_and_format_date(date_str)
            assert result is not None
            assert "2023" in result

    def test_parse_and_format_date_invalid(self):
        """Test date parsing with invalid dates"""
        invalid_dates = [
            "invalid-date",
            "",
            None,
            "123456"
        ]
        
        for date_str in invalid_dates:
            result = self.consumer.parse_and_format_date(date_str)
            assert result is None

    def test_normalize_article_dates(self):
        """Test article date normalization"""
        article = {
            "metadata": {
                "pub_date": "January 1, 2023"
            }
        }
        
        result = self.consumer.normalize_article_dates(article)
        assert "pub_date" in result["metadata"]
        assert "2023" in result["metadata"]["pub_date"]

    def test_normalize_article_dates_invalid(self):
        """Test article date normalization with invalid date"""
        article = {
            "metadata": {
                "pub_date": "invalid-date"
            }
        }
        
        result = self.consumer.normalize_article_dates(article)
        assert "pub_date" not in result["metadata"]

    @patch('news_data_crawling.db.elasticsearch_manager.ElasticsearchManager')
    @patch('news_data_crawling.consumers.news_consumers.NewsConsumer')
    def test_run_consumer_no_messages(self, mock_consumer, mock_es_manager):
        """Test consumer with no messages"""
        mock_consumer_instance = Mock()
        mock_consumer.return_value = mock_consumer_instance
        mock_consumer_instance.poll.return_value = None
        
        mock_es_instance = Mock()
        mock_es_manager.return_value = mock_es_instance
        
        result = self.consumer.run_consumer(limit=1)
        assert result == []
        mock_consumer_instance.subscribe.assert_called_once()

    @patch('news_data_crawling.db.elasticsearch_manager.ElasticsearchManager')
    @patch('news_data_crawling.consumers.news_consumers.NewsConsumer')
    def test_run_consumer_with_messages(self, mock_consumer, mock_es_manager):
        """Test consumer with valid messages"""
        mock_consumer_instance = Mock()
        mock_consumer.return_value = mock_consumer_instance
        
        # Mock message
        mock_message = Mock()
        mock_message.error.return_value = None
        mock_message.value.return_value = json.dumps(self.sample_article).encode('utf-8')
        
        mock_consumer_instance.poll.side_effect = [mock_message, None]
        
        mock_es_instance = Mock()
        mock_es_manager.return_value = mock_es_instance
        
        result = self.consumer.run_consumer(limit=2)
        assert len(result) == 1
        mock_es_instance.index_document.assert_called_once()

class TestNewsProducer:
    """Test NewsProducer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.producer = NewsProducer(
            kafka_broker="localhost:9092",
            topic_name="test_topic"
        )
        
        # Sample test articles
        self.sample_articles = [
            {
                "id": "test-article-1",
                "title": "Test Article 1",
                "content": "Test content 1",
                "timestamp": datetime.now()
            },
            {
                "id": "test-article-2", 
                "title": "Test Article 2",
                "content": "Test content 2",
                "timestamp": datetime.now()
            }
        ]

    @patch('news_data_crawling.producers.news_producer.NewsProducer')
    def test_wait_for_kafka_success(self, mock_admin_client):
        """Test successful Kafka connection wait"""
        mock_admin_instance = Mock()
        mock_admin_client.return_value = mock_admin_instance
        mock_admin_instance.list_topics.return_value = Mock(topics={})
        
        result = self.producer.wait_for_kafka("localhost:9092", max_retries=1)
        assert result is True

    @patch('news_data_crawling.producers.news_producer.NewsProducer')
    def test_wait_for_kafka_failure(self, mock_admin_client):
        """Test failed Kafka connection wait"""
        mock_admin_instance = Mock()
        mock_admin_client.return_value = mock_admin_instance
        mock_admin_instance.list_topics.side_effect = Exception("Connection failed")
        
        result = self.producer.wait_for_kafka("localhost:9092", max_retries=1)
        assert result is False

    @patch('news_data_crawling.producers.news_producer.NewsProducer')
    def test_create_topic_if_not_exists(self, mock_admin_client):
        """Test topic creation"""
        mock_admin_instance = Mock()
        mock_admin_client.return_value = mock_admin_instance
        
        # Mock topic doesn't exist
        mock_metadata = Mock()
        mock_metadata.topics = {}
        mock_admin_instance.list_topics.return_value = mock_metadata
        
        # Mock successful topic creation
        mock_future = Mock()
        mock_admin_instance.create_topics.return_value = {"test_topic": mock_future}
        
        result = self.producer.create_topic_if_not_exists("test_topic", "localhost:9092")
        assert result is True

    @patch('news_data_crawling.producers.news_producer.NewsProducer')
    def test_run_producer_with_articles_success(self, mock_admin_client, mock_producer):
        """Test successful article production"""
        # Mock Kafka admin
        mock_admin_instance = Mock()
        mock_admin_client.return_value = mock_admin_instance
        mock_admin_instance.list_topics.return_value = Mock(topics={"test_topic": {}})
        
        # Mock producer
        mock_producer_instance = Mock()
        mock_producer.return_value = mock_producer_instance
        
        result = self.producer.run_producer_with_articles(self.sample_articles)
        assert result == 2  # Both articles should be sent successfully
        assert mock_producer_instance.produce.call_count == 2

    @patch('news_data_crawling.producers.news_producer.NewsProducer')
    def test_run_producer_with_articles_failure(self, mock_admin_client, mock_producer):
        """Test article production with failures"""
        # Mock Kafka admin
        mock_admin_instance = Mock()
        mock_admin_client.return_value = mock_admin_instance
        mock_admin_instance.list_topics.return_value = Mock(topics={"test_topic": {}})
        
        # Mock producer that fails
        mock_producer_instance = Mock()
        mock_producer.return_value = mock_producer_instance
        mock_producer_instance.produce.side_effect = Exception("Produce failed")
        
        result = self.producer.run_producer_with_articles(self.sample_articles)
        assert result == 0  # No articles should be sent successfully

    def test_delivery_report_success(self):
        """Test successful delivery report"""
        initial_success = self.producer.successful_deliveries
        self.producer.delivery_report(None, Mock())
        assert self.producer.successful_deliveries == initial_success + 1

    def test_delivery_report_failure(self):
        """Test failed delivery report"""
        initial_failed = self.producer.failed_deliveries
        mock_error = Mock()
        mock_error.code.return_value = KafkaError._MSG_TIMED_OUT
        self.producer.delivery_report(mock_error, Mock())
        assert self.producer.failed_deliveries == initial_failed + 1

    @patch('data_crawling.producers_consumers.news_producer.Path')
    def test_save_to_local_file(self, mock_path):
        """Test local file saving"""
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.parent.mkdir.return_value = None
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.producer.save_to_local_file(self.sample_articles)
            mock_file.assert_called_once()
            mock_path_instance.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])