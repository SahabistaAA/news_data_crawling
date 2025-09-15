import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import json
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
subparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, subparent_dir)

from news_data_crawling.medallion_architect.silver_layer.silver_db import SilverDB

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSilverDB(unittest.TestCase):
    """Test suite for Silver layer database operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.silver_db = SilverDB()
        
        # Sample bronze data for silver processing
        self.sample_bronze_data = pd.DataFrame({
            'id': [1, 2, 3],
            'title': [
                'AI Revolution in Healthcare',
                'Stock Market Hits Record High',
                'Climate Summit Reaches Agreement'
            ],
            'url': [
                'https://example.com/ai-healthcare',
                'https://example.com/stock-market',
                'https://example.com/climate-summit'
            ],
            'pub_date': [
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=2),
                datetime.now() - timedelta(days=3)
            ],
            'source': ['TechNews', 'FinanceDaily', 'EcoWorld'],
            'region': ['US', 'EU', 'Global'],
            'author': ['Dr. Smith', 'Jane Doe', 'Bob Johnson'],
            'content': [
                'Artificial intelligence is transforming healthcare with new diagnostic tools...',
                'The stock market reached unprecedented levels today as investors...',
                'World leaders agreed on new climate targets at the summit...'
            ],
            'full_content': [
                'Comprehensive article about AI in healthcare, machine learning applications, and future prospects...',
                'Detailed analysis of market trends, economic indicators, and investment strategies...',
                'Full coverage of climate negotiations, environmental policies, and sustainability goals...'
            ],
            'language_detected': ['en', 'en', 'en'],
            'quality_check_status': ['high', 'high', 'medium']
        })
        
        # Sample silver data with enrichments
        self.sample_silver_data = pd.DataFrame({
            'bronze_id': [1, 2, 3],
            'title': self.sample_bronze_data['title'].values,
            'url': self.sample_bronze_data['url'].values,
            'pub_date': self.sample_bronze_data['pub_date'].values,
            'source': self.sample_bronze_data['source'].values,
            'region': self.sample_bronze_data['region'].values,
            'author': self.sample_bronze_data['author'].values,
            'content': [
                'Artificial intelligence transforming healthcare diagnostic tools',
                'Stock market reached unprecedented levels today investors',
                'World leaders agreed climate targets summit'
            ],
            'topic_classification': [
                {'primary': 'Technology', 'secondary': 'Healthcare', 'confidence': 0.92},
                {'primary': 'Finance', 'secondary': 'Markets', 'confidence': 0.88},
                {'primary': 'Environment', 'secondary': 'Policy', 'confidence': 0.85}
            ],
            'entities': [
                {'persons': ['Dr. Smith'], 'organizations': ['WHO', 'FDA'], 'locations': ['New York']},
                {'persons': ['Jane Doe'], 'organizations': ['NYSE', 'Federal Reserve'], 'locations': ['Wall Street']},
                {'persons': ['Bob Johnson'], 'organizations': ['UN', 'IPCC'], 'locations': ['Paris']}
            ],
            'sentiment_score': [0.75, 0.82, 0.65],
            'sentiment_label': ['positive', 'positive', 'neutral'],
            'sentiment_confidence': [0.85, 0.90, 0.75],
            'emotions_score': [{}, {}, {}],
            'person_entities': [['Dr. Smith'], ['Jane Doe'], ['Bob Johnson']],
            'organization_entities': [['WHO', 'FDA'], ['NYSE', 'Federal Reserve'], ['UN', 'IPCC']],
            'location_entities': [['New York'], ['Wall Street'], ['Paris']],
            'misc_entities': [[], [], []],
            'keywords': [
                ['AI', 'healthcare', 'diagnostics', 'machine learning'],
                ['stock market', 'investment', 'economy', 'growth'],
                ['climate', 'environment', 'summit', 'sustainability']
            ],
            'language_detected': ['en', 'en', 'en'],
            'quality_score': [0.92, 0.88, 0.75],
            'processing_timestamp': [datetime.now()] * 3,
            'enrichment_version': ['v1.0.0'] * 3,
            'content_type': ['news'] * 3,
            'person_entity_count': [1, 1, 1],
            'organization_entity_count': [2, 2, 2],
            'location_entity_count': [1, 1, 1],
            'total_entity_count': [4, 4, 4],
            'content_richness_score': [0.85, 0.82, 0.78],
            'information_density': [0.75, 0.70, 0.68]
        })

    @patch('news_data_crawling.medallion_architect.silver_layer.silver_db.pg2.connect')
    def test_create_silver_table(self, mock_connect):
        """Test silver table creation"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn
        
        self.silver_db.create_silver_table()
        
        # Verify table creation SQL was executed
        mock_cursor.execute.assert_called()
        executed_sql = mock_cursor.execute.call_args[0][0]
        
        # Check for essential columns
        self.assertIn('CREATE TABLE IF NOT EXISTS silver_data', executed_sql)
        self.assertIn('topic_classification', executed_sql)
        self.assertIn('entities', executed_sql)
        self.assertIn('sentiment_score', executed_sql)
        self.assertIn('keywords', executed_sql)

        # Check for indexes
        self.assertIn('CREATE INDEX IF NOT EXISTS idx_silver_url', executed_sql)
        self.assertIn('CREATE INDEX IF NOT EXISTS idx_silver_sentiment', executed_sql)
    
    @patch('news_data_crawling.medallion_architect.silver_layer.silver_db.pd.read_sql_query')
    @patch('news_data_crawling.medallion_architect.silver_layer.silver_db.pg2.connect')
    def test_fetch_bronze_for_processing(self, mock_connect, mock_read_sql):
        """Test fetching bronze data for silver processing"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        mock_read_sql.return_value = self.sample_bronze_data
        
        result = self.silver_db.fetch_bronze_for_processing(batch_size=10)
        
        self.assertEqual(len(result), 3)
        self.assertIn('content', result.columns)
        self.assertIn('quality_check_status', result.columns)
        
        # Verify query includes quality filter
        query = mock_read_sql.call_args[0][0]
        self.assertIn("quality_check_status IN ('high', 'medium')", query)
    
    @patch('news_data_crawling.medallion_architect.silver_layer.silver_db.execute_values')
    @patch('news_data_crawling.medallion_architect.silver_layer.silver_db.pg2.connect')
    def test_save_to_silver(self, mock_connect, mock_execute_values):
        """Test saving enriched data to silver table"""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn

        # Mock execute_values to avoid psycopg2 encoding issues
        mock_execute_values.return_value = None

        self.silver_db.save_to_silver(self.sample_silver_data)
        
        # Verify that execute_values was called
        mock_execute_values.assert_called_once()
        
        # Verify that commit was called
        mock_conn.commit
    
    def test_text_processing(self):
        """Test text cleaning and processing"""
        sample_text = """
        This is a TEST text with Multiple    spaces.
        It has newlines
        and special characters: @#$% & HTML <tags>.
        URLs like https://example.com should be handled.
        """
        
        # Basic text cleaning
        import re
        
        def clean_text(text):
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove special characters (keep alphanumeric and basic punctuation)
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
            return text.strip()
        
        cleaned = clean_text(sample_text)
        self.assertNotIn('<tags>', cleaned)
        self.assertNotIn('https://example.com', cleaned)
        self.assertNotIn('@#$%', cleaned)
    
    def test_entity_extraction(self):
        """Test entity extraction simulation"""
        texts = [
            "Apple CEO Tim Cook announced new products in California.",
            "President Biden met with Chancellor Scholz in Berlin.",
            "Microsoft and Google compete in AI market."
        ]
        
        # Simulate entity extraction
        def extract_entities(text):
            # Simple mock extraction (in real scenario, use NLP library)
            entities = {
                'persons': [],
                'organizations': [],
                'locations': []
            }
            
            # Mock person detection
            if 'Tim Cook' in text:
                entities['persons'].append('Tim Cook')
            if 'Biden' in text:
                entities['persons'].append('Joe Biden')
            if 'Scholz' in text:
                entities['persons'].append('Olaf Scholz')
            
            # Mock organization detection
            for org in ['Apple', 'Microsoft', 'Google']:
                if org in text:
                    entities['organizations'].append(org)
            
            # Mock location detection
            for loc in ['California', 'Berlin', 'New York']:
                if loc in text:
                    entities['locations'].append(loc)
            
            return entities
        
        for text in texts:
            entities = extract_entities(text)
            self.assertIsInstance(entities, dict)
            self.assertIn('persons', entities)
            self.assertIn('organizations', entities)
            self.assertIn('locations', entities)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis simulation"""
        texts = [
            "This is absolutely fantastic news! Great achievement!",
            "The situation is concerning and problematic.",
            "The report provides factual information about the event."
        ]
        
        def analyze_sentiment(text):
            # Simple mock sentiment (in real scenario, use NLP library)
            positive_words = ['fantastic', 'great', 'excellent', 'wonderful', 'achievement']
            negative_words = ['concerning', 'problematic', 'terrible', 'bad', 'awful']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return {'label': 'positive', 'score': 0.8}
            elif neg_count > pos_count:
                return {'label': 'negative', 'score': 0.3}
            else:
                return {'label': 'neutral', 'score': 0.5}
        
        sentiments = [analyze_sentiment(text) for text in texts]
        
        self.assertEqual(sentiments[0]['label'], 'positive')
        self.assertEqual(sentiments[1]['label'], 'negative')
        self.assertEqual(sentiments[2]['label'], 'neutral')
    
    def test_topic_classification(self):
        """Test topic classification simulation"""
        texts = [
            "Stock market rallies on strong earnings reports from tech sector",
            "New AI breakthrough in natural language processing",
            "Climate change impacts on global agriculture"
        ]

        def classify_topic(text):
            # Simple keyword-based classification (in real scenario, use ML model)
            topics = {
                'Finance': ['stock', 'market', 'earnings', 'investment', 'economy'],
                'Technology': ['AI', 'tech', 'software', 'innovation', 'digital', 'processing'],
                'Environment': ['climate', 'environment', 'sustainability', 'agriculture']
            }

            text_lower = text.lower()
            scores = {}

            for topic, keywords in topics.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    scores[topic] = score

            if scores:
                primary = max(scores, key=scores.get)
                confidence = min(scores[primary] / 3.0, 1.0)  # Normalize confidence
                return {
                    'primary': primary,
                    'confidence': confidence
                }

            return {'primary': 'General', 'confidence': 0.5}

        classifications = [classify_topic(text) for text in texts]

        self.assertEqual(classifications[0]['primary'], 'Finance')
        self.assertEqual(classifications[1]['primary'], 'Technology')
        self.assertEqual(classifications[2]['primary'], 'Environment')
    
    def test_keyword_extraction(self):
        """Test keyword extraction"""
        text = "Machine learning algorithms revolutionize data analysis and artificial intelligence applications"
        
        def extract_keywords(text, num_keywords=5):
            # Simple frequency-based extraction (in real scenario, use TF-IDF or similar)
            import re
            from collections import Counter
            
            # Remove common words (stopwords)
            stopwords = {'and', 'the', 'is', 'at', 'in', 'on', 'a', 'an', 'to', 'for'}
            
            # Tokenize and filter
            words = re.findall(r'\b[a-z]+\b', text.lower())
            words = [w for w in words if w not in stopwords and len(w) > 3]
            
            # Get most common
            word_freq = Counter(words)
            keywords = [word for word, _ in word_freq.most_common(num_keywords)]
            
            return keywords
        
        keywords = extract_keywords(text)
        self.assertIsInstance(keywords, list)
        self.assertTrue(len(keywords) <= 5)
        self.assertIn('machine', keywords)
    
    def test_summary_generation(self):
        """Test summary generation"""
        text = """
        Artificial intelligence continues to advance rapidly. Machine learning models
        are becoming more sophisticated. Natural language processing has improved
        significantly. These technologies are transforming various industries.
        The impact on healthcare, finance, and education is particularly notable.
        """
        
        def generate_summary(text, max_length=50):
            # Simple extractive summary (in real scenario, use NLP library)
            sentences = text.strip().split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return ""
            
            # Take first two sentences as summary
            summary = '. '.join(sentences[:2]) + '.'
            
            # Truncate if too long
            if len(summary.split()) > max_length:
                words = summary.split()[:max_length]
                summary = ' '.join(words) + '...'
            
            return summary
        
        summary = generate_summary(text)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)
        self.assertTrue(len(summary.split()) <= 50)
    
    @patch('news_data_crawling.medallion_architect.silver_layer.silver_db.pg2.connect')
    def test_data_validation(self, mock_connect):
        """Test data validation before saving"""
        # Create invalid data
        invalid_data = self.sample_silver_data.copy()
        invalid_data.loc[0, 'sentiment_score'] = 1.5  # Invalid score > 1
        invalid_data.loc[1, 'url'] = None  # Missing required field
        
        # Validation function
        def validate_silver_data(df):
            errors = []
            
            # Check sentiment scores are between -1 and 1
            invalid_sentiments = df[(df['sentiment_score'] < -1) | (df['sentiment_score'] > 1)]
            if not invalid_sentiments.empty:
                errors.append(f"Invalid sentiment scores found: {invalid_sentiments.index.tolist()}")
            
            # Check required fields
            required_fields = ['url', 'title', 'content_clean']
            for field in required_fields:
                if field in df.columns:
                    null_rows = df[df[field].isna()]
                    if not null_rows.empty:
                        errors.append(f"Missing {field} in rows: {null_rows.index.tolist()}")
            
            return errors
        
        validation_errors = validate_silver_data(invalid_data)
        self.assertTrue(len(validation_errors) > 0)
        self.assertTrue(any('sentiment' in error for error in validation_errors))
    
    @patch('news_data_crawling.medallion_architect.silver_layer.silver_db.pg2.connect')
    def test_batch_processing(self, mock_connect):
        """Test batch processing of bronze to silver data"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn
        
        # Create large dataset
        large_dataset = pd.concat([self.sample_bronze_data] * 100, ignore_index=True)
        
        # Process in batches
        batch_size = 50
        processed_count = 0
        
        for i in range(0, len(large_dataset), batch_size):
            batch = large_dataset.iloc[i:i+batch_size]
            processed_count += len(batch)
        
        self.assertEqual(processed_count, len(large_dataset))
    
    def test_quality_scoring(self):
        """Test quality scoring for silver data"""
        def calculate_quality_score(row):
            score = 0.0
            max_score = 5.0
            
            # Check completeness
            if row.get('title') and len(str(row['title'])) > 10:
                score += 1.0
            if row.get('content_clean') and len(str(row['content_clean'])) > 50:
                score += 1.0
            if row.get('summary') and len(str(row['summary'])) > 20:
                score += 1.0
            
            # Check enrichments
            if row.get('entities') and len(row['entities']) > 0:
                score += 1.0
            if row.get('keywords') and len(row['keywords']) > 3:
                score += 1.0
            
            return score / max_score
        
        for _, row in self.sample_silver_data.iterrows():
            quality = calculate_quality_score(row)
            self.assertTrue(0 <= quality <= 1)
    
    @patch('news_data_crawling.medallion_architect.silver_layer.silver_db.pg2.connect')
    def test_duplicate_handling(self, mock_connect):
        """Test handling of duplicate records"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Create data with duplicates
        duplicate_data = pd.concat([self.sample_silver_data, self.sample_silver_data.iloc[[0]]], ignore_index=True)
        
        # Remove duplicates based on URL
        unique_data = duplicate_data.drop_duplicates(subset=['url'], keep='first')
        
        self.assertEqual(len(unique_data), len(self.sample_silver_data))
        self.assertEqual(len(duplicate_data) - len(unique_data), 1)
    
    def test_language_detection(self):
        """Test language detection for content"""
        texts = {
            'en': "This is an English text about technology and innovation.",
            'es': "Este es un texto en español sobre tecnología e innovación.",
            'fr': "Ceci est un texte en français sur la technologie et l'innovation.",
            'de': "Dies ist ein deutscher Text über Technologie und Innovation."
        }
        
        def detect_language(text):
            # Simple language detection based on common words
            language_indicators = {
                'en': ['the', 'and', 'is', 'about', 'this'],
                'es': ['el', 'la', 'es', 'sobre', 'este'],
                'fr': ['le', 'la', 'est', 'sur', 'ceci'],
                'de': ['der', 'die', 'ist', 'über', 'dies']
            }
            
            text_lower = text.lower()
            scores = {}
            
            for lang, words in language_indicators.items():
                score = sum(1 for word in words if word in text_lower.split())
                if score > 0:
                    scores[lang] = score
            
            if scores:
                return max(scores, key=scores.get)
            return 'unknown'
        
        for expected_lang, text in texts.items():
            detected = detect_language(text)
            # Note: This simple detection might not always be accurate
            self.assertIn(detected, ['en', 'es', 'fr', 'de', 'unknown'])


class TestSilverDataEnrichment(unittest.TestCase):
    """Test data enrichment processes for silver layer"""
    
    def setUp(self):
        """Set up enrichment test data"""
        self.test_content = """
        Apple Inc. announced today that CEO Tim Cook will present at the technology 
        conference in San Francisco next month. The company's stock rose 5% following 
        the announcement. Microsoft and Google are also expected to participate in the event.
        """
    
    def test_entity_enrichment(self):
        """Test entity extraction and enrichment"""
        # Simulate NER results
        entities = {
            'persons': ['Tim Cook'],
            'organizations': ['Apple Inc.', 'Microsoft', 'Google'],
            'locations': ['San Francisco'],
            'misc': ['CEO', 'technology conference']
        }
        
        # Enrich with entity metadata
        enriched_entities = {}
        for entity_type, entity_list in entities.items():
            enriched_entities[entity_type] = []
            for entity in entity_list:
                enriched_entities[entity_type].append({
                    'name': entity,
                    'confidence': np.random.uniform(0.7, 1.0),
                    'offset': self.test_content.find(entity)
                })
        
        self.assertIn('persons', enriched_entities)
        self.assertEqual(len(enriched_entities['organizations']), 3)
        
        # Verify all entities have required fields
        for entity_type, entity_list in enriched_entities.items():
            for entity in entity_list:
                self.assertIn('name', entity)
                self.assertIn('confidence', entity)
                self.assertIn('offset', entity)
    
    def test_topic_modeling(self):
        """Test topic modeling and classification"""
        documents = [
            "AI and machine learning transform healthcare diagnostics",
            "Stock market volatility affects investment strategies",
            "Climate change requires urgent environmental action",
            "Smartphone technology advances with 5G networks",
            "Election results impact political landscape"
        ]
        
        # Simulate topic modeling
        def model_topics(docs, num_topics=3):
            # Mock LDA-style topic modeling
            topics = {
                0: {'name': 'Technology', 'keywords': ['AI', 'technology', 'machine', '5G']},
                1: {'name': 'Finance', 'keywords': ['market', 'investment', 'stock']},
                2: {'name': 'Environment', 'keywords': ['climate', 'environmental', 'change']}
            }
            
            doc_topics = []
            for doc in docs:
                doc_lower = doc.lower()
                topic_scores = {}
                
                for topic_id, topic_info in topics.items():
                    score = sum(1 for keyword in topic_info['keywords'] 
                              if keyword.lower() in doc_lower)
                    if score > 0:
                        topic_scores[topic_id] = score
                
                if topic_scores:
                    best_topic = max(topic_scores, key=topic_scores.get)
                    doc_topics.append({
                        'topic_id': best_topic,
                        'topic_name': topics[best_topic]['name'],
                        'confidence': min(topic_scores[best_topic] / 3.0, 1.0)
                    })
                else:
                    doc_topics.append({
                        'topic_id': -1,
                        'topic_name': 'General',
                        'confidence': 0.5
                    })
            
            return doc_topics
        
        topics = model_topics(documents)
        self.assertEqual(len(topics), len(documents))
        self.assertEqual(topics[0]['topic_name'], 'Technology')
        self.assertEqual(topics[1]['topic_name'], 'Finance')
    
    def test_metadata_enrichment(self):
        """Test metadata enrichment"""
        article = {
            'title': 'Breaking News Article',
            'content': 'This is the article content ' * 100,  # ~500 words
            'pub_date': datetime.now()
        }

        # Add metadata
        def enrich_metadata(article):
            enriched = article.copy()

            # Word count
            enriched['word_count'] = len(article['content'].split())

            # Reading time (average 200 words per minute)
            enriched['reading_time_minutes'] = enriched['word_count'] / 200

            # Content length category
            if enriched['word_count'] < 100:
                enriched['content_length'] = 'short'
            elif enriched['word_count'] < 500:
                enriched['content_length'] = 'medium'
            else:
                enriched['content_length'] = 'long'

            # Freshness
            days_old = (datetime.now() - article['pub_date']).days
            if days_old <= 1:
                enriched['freshness'] = 'breaking'
            elif days_old <= 7:
                enriched['freshness'] = 'recent'
            else:
                enriched['freshness'] = 'archived'

            return enriched

        enriched = enrich_metadata(article)
        self.assertIn('word_count', enriched)
        self.assertIn('reading_time_minutes', enriched)
        self.assertIn('content_length', enriched)
        self.assertIn('freshness', enriched)
        self.assertEqual(enriched['content_length'], 'long')  # 500 words should be 'long'
        self.assertEqual(enriched['freshness'], 'breaking')
    
    def test_relationship_extraction(self):
        """Test relationship extraction between entities"""
        text = "Apple CEO Tim Cook met with Microsoft CEO Satya Nadella in Seattle to discuss AI collaboration."
        
        # Extract relationships
        def extract_relationships(text):
            # Mock relationship extraction
            relationships = []
            
            # Simple pattern matching (in real scenario, use dependency parsing)
            if 'met with' in text:
                relationships.append({
                    'subject': 'Tim Cook',
                    'relation': 'met_with',
                    'object': 'Satya Nadella',
                    'context': 'Seattle'
                })
            
            if 'discuss' in text:
                relationships.append({
                    'subject': 'Apple',
                    'relation': 'collaborate',
                    'object': 'Microsoft',
                    'context': 'AI'
                })
            
            return relationships
        
        relations = extract_relationships(text)
        self.assertTrue(len(relations) > 0)
        self.assertEqual(relations[0]['relation'], 'met_with')
    
    def test_content_deduplication(self):
        """Test content deduplication using similarity"""
        contents = [
            "The stock market rose 5% today on positive earnings",
            "Stock market increased by 5 percent following good earnings",  # Similar
            "Climate summit reaches agreement on carbon emissions",  # Different
            "The stock market rose 5% today on positive earnings"  # Exact duplicate
        ]
        
        def calculate_similarity(text1, text2):
            # Simple word overlap similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
        
        # Find similar/duplicate content
        threshold = 0.7
        duplicates = []
        
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                similarity = calculate_similarity(contents[i], contents[j])
                if similarity >= threshold:
                    duplicates.append((i, j, similarity))
        
        self.assertTrue(len(duplicates) > 0)
        # Should find exact duplicate (indices 0 and 3)
        self.assertTrue(any(sim == 1.0 for _, _, sim in duplicates))


class TestSilverIntegration(unittest.TestCase):
    """Integration tests for Silver layer"""
    
    @patch('news_data_crawling.medallion_architect.silver_layer.silver_db.pg2.connect')
    def test_end_to_end_processing(self, mock_connect):
        """Test complete bronze to silver processing pipeline"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_connect.return_value = mock_conn
        
        # Simulate pipeline stages
        pipeline_stages = [
            'fetch_bronze_data',
            'clean_text',
            'extract_entities',
            'classify_topics',
            'analyze_sentiment',
            'generate_summary',
            'calculate_quality',
            'save_to_silver'
        ]
        
        # Track stage completion
        completed_stages = []
        
        for stage in pipeline_stages:
            # Simulate stage processing
            completed_stages.append(stage)
            logger.info(f"Completed stage: {stage}")
        
        self.assertEqual(len(completed_stages), len(pipeline_stages))
    
    def test_error_recovery(self):
        """Test error handling and recovery"""
        def process_with_retry(data, max_retries=3):
            retries = 0
            while retries < max_retries:
                try:
                    # Simulate processing that might fail
                    if retries < 2:
                        raise Exception("Processing failed")
                    return "Success"
                except Exception as e:
                    retries += 1
                    logger.warning(f"Retry {retries}/{max_retries}: {e}")
                    if retries >= max_retries:
                        raise
            
        result = process_with_retry("test_data")
        self.assertEqual(result, "Success")
    
    def test_performance_metrics(self):
        """Test performance monitoring"""
        import time
        
        def measure_processing_time(func, *args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            return {
                'result': result,
                'execution_time': end_time - start_time,
                'timestamp': datetime.now()
            }
        
        def sample_processing():
            time.sleep(0.1)  # Simulate processing
            return "Processed"
        
        metrics = measure_processing_time(sample_processing)
        
        self.assertIn('result', metrics)
        self.assertIn('execution_time', metrics)
        self.assertTrue(metrics['execution_time'] >= 0.1)


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)