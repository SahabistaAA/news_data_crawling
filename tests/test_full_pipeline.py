import unittest
from unittest.mock import patch, MagicMock
from airflow.models import DagBag
from producers.news_producers import NewsProducer
from consumers.news_consumers import NewsConsumer
import pandas as pd
from medallion_architect.bronze_layer.cleaning import DataCleaner
from medallion_architect.silver_layer.silver_enrichment import SilverProcessor
from db.postgres_manager import PostgresManager
from db.elasticsearch_manager import ElasticsearchManager

class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        # Load all DAGs for testing
        self.dagbag = DagBag()

    def test_dags_loaded(self):
        """Test that all DAGs are loaded without errors"""
        self.assertFalse(
            len(self.dagbag.import_errors),
            f"DAG import failures. Errors: {self.dagbag.import_errors}"
        )

    @patch('db.postgres_manager.PostgresManager.save_raw_articles')
    @patch('db.elasticsearch_manager.ElasticsearchManager.index_document')
    @patch('producers.news_producers.NewsProducer.run_producer_with_articles')
    @patch('consumers.news_consumers.NewsConsumer.run_consumer')
    def test_full_pipeline_flow(self, mock_consumer_run, mock_producer_run, mock_es_index, mock_pg_save_raw):
        """
        Test the complete pipeline flow:
        Airflow Scheduling >> Data Crawling >> Kafka Producer >> Kafka Consumer >> Elasticsearch (mapped) + postgresql (raw)
        Then: PostgreSQL (raw) >> bronze_layer >> silver layer
        """
        # Mock producer to simulate producing articles
        mock_producer_run.return_value = 2

        # Mock consumer to simulate consuming articles
        consumed_articles = [
            {
                "id": "1",
                "title": "Test Article 1",
                "content": "Content 1",
                "url": "https://example.com/1",
                "pub_date": "2024-01-01T00:00:00Z",
                "source": "Test Source",
                "author": "Test Author",
                "processed_at": "2024-01-01T01:00:00Z"
            },
            {
                "id": "2",
                "title": "Test Article 2",
                "content": "Content 2",
                "url": "https://example.com/2",
                "pub_date": "2024-01-01T02:00:00Z",
                "source": "Test Source",
                "author": "Test Author 2",
                "processed_at": "2024-01-01T03:00:00Z"
            },
        ]
        mock_consumer_run.return_value = consumed_articles

        # Mock Elasticsearch indexing
        mock_es_index.return_value = True

        # Mock PostgreSQL raw data save
        mock_pg_save_raw.return_value = 2

        # Step 1: Simulate Airflow Scheduling >> Data Crawling >> Kafka Producer >> Kafka Consumer
        producer = NewsProducer()
        consumer = NewsConsumer()

        # Simulate producing articles
        articles = [
            {"id": "1", "title": "Test Article 1", "content": "Content 1"},
            {"id": "2", "title": "Test Article 2", "content": "Content 2"},
        ]
        produced_count = producer.run_producer_with_articles(articles)
        self.assertEqual(produced_count, 2)

        # Simulate consuming articles
        consumed_articles_result = consumer.run_consumer(limit=2)
        self.assertEqual(len(consumed_articles_result), 2)

        # Step 2: Simulate storage to Elasticsearch and PostgreSQL raw
        es_manager = ElasticsearchManager()
        pg_manager = PostgresManager()

        # Simulate saving to Elasticsearch
        for article in consumed_articles_result:
            es_manager.index_document("news", article)

        # Simulate saving raw data to PostgreSQL
        saved_count = pg_manager.save_raw_articles(consumed_articles_result)
        self.assertEqual(saved_count, 2)

        # Step 3: Simulate PostgreSQL (raw) >> bronze_layer >> silver layer
        # Bronze layer processing
        cleaner = DataCleaner()
        df = pd.DataFrame(consumed_articles_result)
        cleaned_df = cleaner.clean_data(df)
        self.assertFalse(cleaned_df.empty)
        self.assertIn('quality_check_status', cleaned_df.columns)

        # Simulate saving bronze data to PostgreSQL
        bronze_articles = cleaned_df.to_dict('records')
        with patch('db.postgres_manager.PostgresManager.save_bronze_articles') as mock_save_bronze:
            mock_save_bronze.return_value = len(bronze_articles)
            saved_bronze_count = pg_manager.save_bronze_articles(bronze_articles)
            self.assertEqual(saved_bronze_count, len(bronze_articles))

        # Silver layer processing
        silver_processor = SilverProcessor()
        with patch.object(silver_processor, '_initialize_models'), \
             patch.object(silver_processor, 'db') as mock_silver_db:

            # Mock silver database operations
            mock_silver_db.create_silver_table.return_value = True
            mock_silver_db.save_to_silver.return_value = True

            # Process bronze to silver
            silver_processor.process_bronze_to_silver(cleaned_df)

            # Verify silver processing was called
            mock_silver_db.create_silver_table.assert_called_once()
            mock_silver_db.save_to_silver.assert_called()

        # Verify the complete pipeline flow
        self.assertEqual(len(consumed_articles_result), 2)
        self.assertEqual(len(cleaned_df), 2)
        mock_producer_run.assert_called_once()
        mock_consumer_run.assert_called_once()
        mock_es_index.assert_called()
        mock_pg_save_raw.assert_called_once()

if __name__ == "__main__":
    unittest.main()
