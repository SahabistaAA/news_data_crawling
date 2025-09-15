import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from .sentiment_analysis import SentimentAnalyzer
from .ner import EntityExtractor
from .knowledge_graph import KnowledgeGraphBuilder
from .topic_classification import TopicClassifier, KeywordExtractor
from ..database.silver_db import SilverDB
from datetime import datetime

logger = logging.getLogger(__name__)


class SilverProcessor:
    """Main processor for silver layer operations"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.entity_extractor = EntityExtractor()
        self.topic_classifier = TopicClassifier()
        self.keyword_extractor = KeywordExtractor()
        self.kg_builder = KnowledgeGraphBuilder()
        self.db = SilverDB()
        self.enrichment_version = "v1.0.0"
    
    def process_bronze_to_silver(self, bronze_data: pd.DataFrame) -> None:
        """
        Main pipeline to process bronze data into silver layer
        """
        try:
            logger.info(f"Starting silver layer processing for {len(bronze_data)} records")
            
            # Create silver table if it doesn't exist
            self.db.create_silver_table()
            
            # Initialize ML models
            self._initialize_models()
            
            # Process data in batches for memory efficiency
            batch_size = 100
            total_processed = 0
            
            for i in range(0, len(bronze_data), batch_size):
                batch = bronze_data.iloc[i:i + batch_size]
                enriched_batch = self._process_batch(batch)
                
                # Save to silver database
                self.db.save_to_silver(enriched_batch)
                
                # Save to knowledge graph
                self.kg_builder.create_knowledge_graph(enriched_batch)
                
                total_processed += len(enriched_batch)
                logger.info(f"Processed batch {i//batch_size + 1}: {total_processed}/{len(bronze_data)} records")
            
            logger.info("Silver layer processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in silver layer processing: {e}")
            raise
    
    def _initialize_models(self) -> None:
        """Initialize ML models for processing"""
        try:
            logger.info("Initializing ML models...")
            self.sentiment_analyzer.load_models()
            self.entity_extractor.load_models()
            logger.info("ML models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of bronze data"""
        try:
            enriched_batch = batch.copy()
            
            # Add processing timestamp
            enriched_batch['processing_timestamp'] = pd.Timestamp.now()
            enriched_batch['enrichment_version'] = self.enrichment_version
            
            # Apply enrichment operations
            enriched_batch = self._apply_sentiment_analysis(enriched_batch)
            enriched_batch = self._apply_entity_extraction(enriched_batch)
            enriched_batch = self._apply_topic_classification(enriched_batch)
            enriched_batch = self._apply_keyword_extraction(enriched_batch)
            enriched_batch = self._calculate_enrichment_metrics(enriched_batch)
            
            return enriched_batch
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
    
    def _apply_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sentiment analysis to the batch"""
        try:
            logger.debug(f"Applying sentiment analysis to {len(df)} records")
            
            sentiment_results = []
            
            for _, row in df.iterrows():
                content = str(row.get('content', ''))
                language = str(row.get('language_detected', 'english'))
                
                # Perform sentiment analysis
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(content, language)
                sentiment_results.append(sentiment_result)
            
            # Add sentiment results to dataframe
            df['sentiment_score'] = [r.get('score', 0.0) for r in sentiment_results]
            df['sentiment_label'] = [r.get('label', 'neutral') for r in sentiment_results]
            df['sentiment_confidence'] = [r.get('confidence', 0.0) for r in sentiment_results]
            df['emotions_score'] = [r.get('emotion_scores', {}) for r in sentiment_results]
            
            logger.debug("Sentiment analysis completed")
            return df
            
        except Exception as e:
            logger.error(f"Error applying sentiment analysis: {e}")
            raise
    
    def _apply_entity_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply named entity recognition to the batch"""
        try:
            logger.debug(f"Applying entity extraction to {len(df)} records")
            
            entity_results = []
            
            for _, row in df.iterrows():
                content = str(row.get('content', ''))
                title = str(row.get('title', ''))
                language = str(row.get('language_detected', 'english'))
                
                # Combine title and content for better entity extraction
                text_to_analyze = f"{title}. {content}"
                
                # Perform entity extraction
                entity_result = self.entity_extractor.extract_entities(text_to_analyze, language)
                entity_results.append(entity_result)
            
            # Add entity results to dataframe
            df['entities'] = [r.get('entities', []) for r in entity_results]
            df['person_entities'] = [r.get('person_entities', []) for r in entity_results]
            df['organization_entities'] = [r.get('organization_entities', []) for r in entity_results]
            df['location_entities'] = [r.get('location_entities', []) for r in entity_results]
            df['misc_entities'] = [r.get('misc_entities', []) for r in entity_results]
            
            logger.debug("Entity extraction completed")
            return df
            
        except Exception as e:
            logger.error(f"Error applying entity extraction: {e}")
            raise

    def _apply_topic_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply topic classification to the batch"""
        try:
            logger.debug(f"Applying topic classification to {len(df)} records")

            topic_results = []

            for _, row in df.iterrows():
                content = str(row.get('content', ''))
                title = str(row.get('title', ''))

                # Perform topic classification
                topic_result = self.topic_classifier.classify_topic(content, title)
                topic_results.append(topic_result)

            # Add topic results to dataframe
            df['topic_classification'] = topic_results

            logger.debug("Topic classification completed")
            return df

        except Exception as e:
            logger.error(f"Error applying topic classification: {e}")
            raise

    def _apply_keyword_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply keyword extraction to the batch"""
        try:
            logger.debug(f"Applying keyword extraction to {len(df)} records")

            keyword_results = []

            for _, row in df.iterrows():
                content = str(row.get('content', ''))
                title = str(row.get('title', ''))

                # Perform keyword extraction
                keywords = self.keyword_extractor.extract_keywords(content, title)
                keyword_results.append(keywords)

            # Add keyword results to dataframe
            df['keywords'] = keyword_results

            logger.debug("Keyword extraction completed")
            return df

        except Exception as e:
            logger.error(f"Error applying keyword extraction: {e}")
            raise

    def _calculate_enrichment_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional metrics for enriched data"""
        try:
            # Entity counts
            df['person_entity_count'] = df['person_entities'].apply(len)
            df['organization_entity_count'] = df['organization_entities'].apply(len)
            df['location_entity_count'] = df['location_entities'].apply(len)
            df['total_entity_count'] = (
                df['person_entity_count'] + 
                df['organization_entity_count'] + 
                df['location_entity_count']
            )
            
            # Content richness score
            df['content_richness_score'] = df.apply(self._calculate_content_richness, axis=1)
            
            # Information density score
            df['information_density'] = df.apply(self._calculate_information_density, axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating enrichment metrics: {e}")
            raise
    
    def _calculate_content_richness(self, row) -> float:
        """Calculate content richness score based on various factors"""
        try:
            score = 0.0
            
            # Entity diversity (0.4 weight)
            entity_types = 0
            if len(row.get('person_entities', [])) > 0:
                entity_types += 1
            if len(row.get('organization_entities', [])) > 0:
                entity_types += 1
            if len(row.get('location_entities', [])) > 0:
                entity_types += 1
            
            entity_diversity_score = min(entity_types / 3.0, 1.0)
            score += entity_diversity_score * 0.4
            
            # Sentiment confidence (0.2 weight)
            sentiment_confidence = float(row.get('sentiment_confidence', 0.0))
            score += sentiment_confidence * 0.2
            
            # Content length factor (0.2 weight)
            content_length = len(str(row.get('content', '')))
            if content_length > 1000:
                length_score = 1.0
            elif content_length > 500:
                length_score = 0.8
            elif content_length > 200:
                length_score = 0.6
            else:
                length_score = 0.3
            
            score += length_score * 0.2
            
            # Author presence (0.1 weight)
            author = str(row.get('author', ''))
            if author and author != 'nan' and len(author) > 2:
                score += 0.1
            
            # Quality score from bronze layer (0.05 weight)
            quality_score = float(row.get('quality_score', 0.0))
            score += quality_score * 0.05

            # Topic classification confidence (0.05 weight)
            topic_classification = row.get('topic_classification', {})
            if isinstance(topic_classification, dict):
                topic_confidence = float(topic_classification.get('confidence', 0.0))
                score += topic_confidence * 0.05

            # Keywords presence (0.05 weight)
            keywords = row.get('keywords', [])
            if keywords and len(keywords) > 0:
                keyword_score = min(len(keywords) / 10.0, 1.0)  # Normalize to 0-1
                score += keyword_score * 0.05

            return round(min(score, 1.0), 3)
            
        except Exception as e:
            logger.error(f"Error calculating content richness: {e}")
            return 0.0
    
    def _calculate_information_density(self, row) -> float:
        """Calculate information density score"""
        try:
            content = str(row.get('content', ''))
            if not content:
                return 0.0
            
            word_count = len(content.split())
            entity_count = int(row.get('total_entity_count', 0))
            
            if word_count == 0:
                return 0.0
            
            # Calculate entities per 100 words
            density = (entity_count / word_count) * 100
            
            # Normalize to 0-1 scale (assuming 5 entities per 100 words as good density)
            normalized_density = min(density / 5.0, 1.0)
            
            return round(normalized_density, 3)
            
        except Exception as e:
            logger.error(f"Error calculating information density: {e}")
            return 0.0
    
    def get_silver_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve silver data for gold layer processing"""
        try:
            return self.db.fetch_silver_data(limit=limit)
        except Exception as e:
            logger.error(f"Error fetching silver data: {e}")
            raise
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            return self.db.get_silver_stats()
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            raise