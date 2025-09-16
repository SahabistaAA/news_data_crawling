import unittest
from unittest.mock import patch, MagicMock
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta

# Import the DAGs to test
from airflow.dags.news_pipeline_dag import dag as news_pipeline_dag
from airflow.dags.crawl_news_dag import dag as crawl_news_dag
from airflow.dags.produce_news_dag import dag as produce_news_dag

class TestNewsPipelineDAG(unittest.TestCase):
    def test_news_pipeline_dag_structure(self):
        """Test that the main news pipeline DAG has the correct structure"""
        self.assertIsInstance(news_pipeline_dag, DAG)
        self.assertEqual(news_pipeline_dag.dag_id, "news_pipeline_dag")
        self.assertEqual(news_pipeline_dag.schedule_interval, "@hourly")

        # Check that task groups exist
        task_group_ids = [tg.task_group_id for tg in news_pipeline_dag.task_groups]
        self.assertIn("crawl", task_group_ids)
        self.assertIn("produce", task_group_ids)
        self.assertIn("consume", task_group_ids)
        self.assertIn("store", task_group_ids)

        # Check start and end tasks
        self.assertIn("start", news_pipeline_dag.task_dict)
        self.assertIn("end", news_pipeline_dag.task_dict)

    def test_crawl_news_dag_structure(self):
        """Test that the crawl news DAG has the correct structure"""
        self.assertIsInstance(crawl_news_dag, DAG)
        self.assertEqual(crawl_news_dag.dag_id, "crawl_news_dag")
        self.assertEqual(crawl_news_dag.schedule_interval, "@hourly")

        # Check crawl_news task exists
        self.assertIn("crawl_news", crawl_news_dag.task_dict)
        crawl_task = crawl_news_dag.task_dict["crawl_news"]
        self.assertIsInstance(crawl_task, PythonOperator)

    def test_produce_news_dag_structure(self):
        """Test that the produce news DAG has the correct structure"""
        self.assertIsInstance(produce_news_dag, DAG)
        self.assertEqual(produce_news_dag.dag_id, "produce_news_dag")
        self.assertEqual(produce_news_dag.schedule_interval, "@hourly")

        # Check tasks exist
        self.assertIn("wait_for_crawl", produce_news_dag.task_dict)
        self.assertIn("produce_news", produce_news_dag.task_dict)

    @patch('airflow.dags.crawl_news_dag.NewsCrawler')
    def test_crawl_rss_task_execution(self, mock_crawler_class):
        """Test the crawl_rss_task function execution"""
        from airflow.dags.crawl_news_dag import crawl_rss_task

        mock_crawler = MagicMock()
        mock_crawler.crawl_rss.return_value = [{"id": "1", "title": "Test Article"}]
        mock_crawler_class.return_value = mock_crawler

        mock_ti = MagicMock()
        kwargs = {'ti': mock_ti}

        crawl_rss_task(**kwargs)

        mock_crawler.crawl_rss.assert_called_once()
        mock_ti.xcom_push.assert_called_once_with(key="raw_articles", value=[{"id": "1", "title": "Test Article"}])

    @patch('airflow.dags.produce_news_dag.KafkaProducerOperator')
    def test_produce_news_task_execution(self, mock_producer_class):
        """Test the produce_news_task function execution"""
        from airflow.dags.produce_news_dag import produce_news_task

        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer

        articles = [{"id": "1", "title": "Test Article"}]
        kwargs = {}

        produce_news_task(articles, **kwargs)

        mock_producer_class.assert_called_once_with(topic="news_topic")
        mock_producer.produce_messages.assert_called_once_with(articles)

class TestDAGDependencies(unittest.TestCase):
    def test_news_pipeline_dependencies(self):
        """Test that the main pipeline has correct task dependencies"""
        dag = news_pipeline_dag

        # Get task groups
        crawl_group = next(tg for tg in dag.task_groups if tg.task_group_id == "crawl")
        produce_group = next(tg for tg in dag.task_groups if tg.task_group_id == "produce")
        consume_group = next(tg for tg in dag.task_groups if tg.task_group_id == "consume")
        store_group = next(tg for tg in dag.task_groups if tg.task_group_id == "store")

        start_task = dag.task_dict["start"]
        end_task = dag.task_dict["end"]

        # Check upstream/downstream relationships
        self.assertIn(start_task, crawl_group.upstream_task_ids)
        self.assertIn(crawl_group, produce_group.upstream_task_ids)
        self.assertIn(produce_group, consume_group.upstream_task_ids)
        self.assertIn(consume_group, store_group.upstream_task_ids)
        self.assertIn(store_group, end_task.upstream_task_ids)

if __name__ == "__main__":
    unittest.main()
