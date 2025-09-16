import unittest
from unittest.mock import patch, MagicMock
from producers.news_producers import NewsProducer
from consumers.news_consumers import NewsConsumer

class TestNewsProducer(unittest.TestCase):
    @patch('producers.news_producers.Producer')
    def test_run_producer_with_articles_success(self, mock_producer_class):
        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer

        producer = NewsProducer()
        articles = [
            {"id": "1", "title": "Test Article 1", "content": "Content 1"},
            {"id": "2", "title": "Test Article 2", "content": "Content 2"},
        ]

        sent_count = producer.run_producer_with_articles(articles)
        self.assertEqual(sent_count, len(articles))
        self.assertEqual(producer.successful_deliveries, len(articles))
        self.assertEqual(producer.failed_deliveries, 0)
        mock_producer.produce.assert_called()
        mock_producer.flush.assert_called()

    @patch('producers.news_producers.Producer')
    def test_run_producer_with_empty_articles(self, mock_producer_class):
        producer = NewsProducer()
        sent_count = producer.run_producer_with_articles([])
        self.assertEqual(sent_count, 0)

class TestNewsConsumer(unittest.TestCase):
    @patch('consumers.news_consumers.ElasticsearchManager')
    @patch('consumers.news_consumers.Consumer')
    def test_run_consumer_processes_messages(self, mock_consumer_class, mock_es_manager_class):
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer
        mock_es_manager = MagicMock()
        mock_es_manager_class.return_value = mock_es_manager

        # Mock messages returned by consumer.poll
        mock_msg1 = MagicMock()
        mock_msg1.value.return_value = b'{"id": "1", "title": "Test Article"}'
        mock_msg1.error.return_value = None

        mock_msg2 = MagicMock()
        mock_msg2.value.return_value = b'{"id": "2", "title": "Test Article 2"}'
        mock_msg2.error.return_value = None

        # After two messages, return None to stop
        mock_consumer.poll.side_effect = [mock_msg1, mock_msg2, None, KeyboardInterrupt()]

        consumer = NewsConsumer()
        articles = consumer.run_consumer(limit=2)

        self.assertEqual(len(articles), 2)
        mock_es_manager.index_document.assert_called()
        mock_consumer.close.assert_called()

if __name__ == "__main__":
    unittest.main()
