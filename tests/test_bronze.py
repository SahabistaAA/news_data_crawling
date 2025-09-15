import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
subparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, subparent_dir)

from news_data_crawling.medallion_architect.bronze_layer.bronze_db import BronzeDB

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBronzeDB(unittest.TestCase):
    """Test suite for Bronze layer database operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bronze_db = BronzeDB()
        
        # Sample test data
        self.sample_raw_data = pd.DataFrame({
            'title': [
                'Breaking News: Tech Innovation',
                'Market Update: Stock Rally',
                'Climate Change Summit 2025'
            ],
            'url': [
                'https://example.com/tech-innovation',
                'https://example.com/market-update',
                'https://example.com/climate-summit'
            ],
            'pub_date': [
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=2),
                datetime.now() - timedelta(days=3)
            ],
            'source': ['TechNews', 'FinanceDaily', 'EcoWorld'],
            'region': ['US', 'EU', 'Global'],
            'author': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'content': [
                'Short tech content...',
                'Short finance content...',
                'Short climate content...'
            ],
            'full_content': [
                'Full article about tech innovation and breakthrough...',
                'Comprehensive market analysis and stock performance...',
                'Detailed coverage of climate summit and agreements...'
            ],
            'ingested_at': [datetime.now()] * 3
        })
        
        self.sample_bronze_data = self.sample_raw_data.copy()
        self.sample_bronze_data['cleaned_at'] = datetime.now()
        self.sample_bronze_data['is_duplicate'] = False
        self.sample_bronze_data['quality_check_status'] = 'high'
    
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pg2.connect')
    def test_connection_manager(self, mock_connect):
        """Test database connection context manager"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        with self.bronze_db.get_connection() as conn:
            self.assertEqual(conn, mock_conn)
        
        mock_conn.close.assert_called_once()
    
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pg2.connect')
    def test_connection_error_handling(self, mock_connect):
        """Test connection error handling"""
        mock_connect.side_effect = psycopg2.OperationalError("Connection failed")
        
        with self.assertRaises(psycopg2.OperationalError):
            with self.bronze_db.get_connection() as conn:
                pass
    
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pg2.connect')
    def test_create_bronze_table(self, mock_connect):
        """Test bronze table creation"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn
        
        self.bronze_db.create_bronze_table()
        
        # Verify table creation SQL was executed
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
        
        # Check if proper indexes were created
        executed_sql = mock_cursor.execute.call_args[0][0]
        self.assertIn('CREATE TABLE IF NOT EXISTS bronze_data', executed_sql)
        self.assertIn('CREATE INDEX IF NOT EXISTS idx_bronze_url', executed_sql)
        self.assertIn('CREATE INDEX IF NOT EXISTS idx_bronze_source', executed_sql)
    
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pd.read_sql_query')
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pg2.connect')
    def test_fetch_raw_data(self, mock_connect, mock_read_sql):
        """Test fetching raw data"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        mock_read_sql.return_value = self.sample_raw_data
        
        # Test without limit
        result = self.bronze_db.fetch_raw_data()
        self.assertEqual(len(result), 3)
        self.assertIn('title', result.columns)
        self.assertIn('url', result.columns)
        
        # Test with limit
        result = self.bronze_db.fetch_raw_data(limit=2)
        mock_read_sql.assert_called()
        query = mock_read_sql.call_args[0][0]
        self.assertIn('LIMIT 2', query)
    
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.execute_values')
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pg2.connect')
    def test_save_to_bronze(self, mock_connect, mock_execute_values):
        """Test saving data to bronze table"""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        self.bronze_db.save_to_bronze(self.sample_bronze_data)
        
        # Verify that execute_values was called with the correct arguments
        mock_execute_values.assert_called_once()
        
        # Verify that commit was called
        mock_conn.commit.assert_called_once()
    
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pg2.connect')
    def test_save_empty_dataframe(self, mock_connect):
        """Test handling empty dataframe"""
        empty_df = pd.DataFrame()
        
        # Should not raise error and should log warning
        with patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.logger.warning') as mock_warning:
            self.bronze_db.save_to_bronze(empty_df)
            mock_warning.assert_called_with("No data to save to bronze table")
    
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pd.read_sql_query')
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pg2.connect')
    def test_fetch_bronze_data(self, mock_connect, mock_read_sql):
        """Test fetching bronze data for silver processing"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        expected_df = self.sample_bronze_data.copy()
        expected_df['language_detected'] = 'en'
        expected_df['content_type'] = 'article'
        mock_read_sql.return_value = expected_df
        
        # Test without filters
        result = self.bronze_db.fetch_bronze_data()
        self.assertEqual(len(result), 3)
        
        # Test with quality filter
        result = self.bronze_db.fetch_bronze_data(quality_filter='high')
        query = mock_read_sql.call_args[0][0]
        self.assertIn("quality_check_status = 'high'", query)
        
        # Test with limit
        result = self.bronze_db.fetch_bronze_data(limit=5)
        query = mock_read_sql.call_args[0][0]
        self.assertIn('LIMIT 5', query)
    
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pg2.connect')
    def test_mark_as_processed(self, mock_connect):
        """Test marking records as processed"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn
        
        record_ids = [1, 2, 3, 4, 5]
        self.bronze_db.mark_as_processed(record_ids)
        
        # Verify UPDATE query was executed
        mock_cursor.execute.assert_called()
        query = mock_cursor.execute.call_args[0][0]
        self.assertIn('UPDATE bronze_data', query)
        self.assertIn('WHERE id IN', query)
        
        # Test with empty list
        self.bronze_db.mark_as_processed([])
        # Should return early without executing query
    
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pg2.connect')
    def test_cleanup_old_records(self, mock_connect):
        """Test cleanup of old records"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 10  # Simulate 10 deleted records
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn
        
        deleted_count = self.bronze_db.cleanup_old_records(days_to_keep=30)
        
        self.assertEqual(deleted_count, 10)
        mock_cursor.execute.assert_called()
        query = mock_cursor.execute.call_args[0][0]
        self.assertIn('DELETE FROM bronze_data', query)
        self.assertIn("INTERVAL '30 days'", query)
        self.assertIn("quality_check_status = 'low'", query)
    
    def test_data_quality_checks(self):
        """Test data quality validation"""
        # Test data with missing required fields
        invalid_data = pd.DataFrame({
            'title': [None, 'Valid Title'],
            'url': ['https://example.com/1', None],
            'content': ['Some content', 'More content']
        })
        
        # Add cleaned_at and quality fields
        invalid_data['cleaned_at'] = datetime.now()
        invalid_data['is_duplicate'] = False
        invalid_data['quality_check_status'] = 'low'
        
        # Check for None values in critical fields
        self.assertTrue(invalid_data['title'].isna().any())
        self.assertTrue(invalid_data['url'].isna().any())
    
    def test_duplicate_detection(self):
        """Test duplicate URL detection logic"""
        data_with_duplicates = pd.DataFrame({
            'url': [
                'https://example.com/article1',
                'https://example.com/article1',  # Duplicate
                'https://example.com/article2'
            ],
            'title': ['Article 1', 'Article 1 Copy', 'Article 2'],
            'content': ['Content 1', 'Content 1', 'Content 2']
        })
        
        # Check for duplicates
        duplicates = data_with_duplicates.duplicated(subset=['url'], keep='first')
        self.assertTrue(duplicates.any())
        self.assertEqual(duplicates.sum(), 1)
    
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.execute_values')
    @patch('news_data_crawling.medallion_architect.bronze_layer.bronze_db.pg2.connect')
    def test_transaction_rollback(self, mock_connect, mock_execute_values):
        """Test transaction rollback on error"""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        # Make execute_values raise an error
        mock_execute_values.side_effect = psycopg2.DatabaseError("Insert failed")
        
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        with self.assertRaises(psycopg2.DatabaseError):
            self.bronze_db.save_to_bronze(self.sample_bronze_data)
        
        # Verify that rollback was called
        mock_conn.rollback.assert_called_once()


class TestBronzeDataProcessing(unittest.TestCase):
    """Test bronze layer data processing logic"""
    
    def setUp(self):
        """Set up test data for processing tests"""
        self.sample_data = pd.DataFrame({
            'title': [
                'Test Article 1',
                'Test Article 2',
                '<h1>HTML Title</h1>',  # HTML content
                'Special Characters: © ® ™'
            ],
            'content': [
                'Normal content.',
                'Content with\nmultiple\nlines.',
                '<p>HTML content</p>',
                'Content with émojis 😀 and symbols'
            ],
            'pub_date': [
                '2024-01-15 10:30:00',
                '2024-01-16 14:45:00',
                'invalid_date',
                None
            ]
        })
    
    def test_text_cleaning(self):
        """Test text cleaning operations"""
        # Remove HTML tags
        import re
        
        def clean_html(text):
            if pd.isna(text):
                return text
            clean = re.compile('<.*?>')
            return re.sub(clean, '', text)
        
        cleaned_titles = self.sample_data['title'].apply(clean_html)
        self.assertEqual(cleaned_titles[2], 'HTML Title')
        
        cleaned_content = self.sample_data['content'].apply(clean_html)
        self.assertEqual(cleaned_content[2], 'HTML content')
    
    def test_date_parsing(self):
        """Test date parsing and validation"""
        dates = pd.to_datetime(self.sample_data['pub_date'], errors='coerce')
        
        # Check valid dates were parsed
        self.assertFalse(pd.isna(dates[0]))
        self.assertFalse(pd.isna(dates[1]))
        
        # Check invalid dates became NaT
        self.assertTrue(pd.isna(dates[2]))
        self.assertTrue(pd.isna(dates[3]))
    
    def test_data_enrichment(self):
        """Test data enrichment operations"""
        df = self.sample_data.copy()
        
        # Add processing metadata
        df['cleaned_at'] = datetime.now()
        df['processing_version'] = '1.0'
        df['quality_score'] = np.random.uniform(0.5, 1.0, len(df))
        
        self.assertIn('cleaned_at', df.columns)
        self.assertIn('processing_version', df.columns)
        self.assertTrue(all(0.5 <= score <= 1.0 for score in df['quality_score']))


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)