import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from collections import Counter

logger = logging.getLogger(__name__)


class DataAggregator:
    """Handles data aggregation for gold layer"""
    
    def __init__(self):
        self.aggregation_functions = {
            'sentiment_score': ['mean', 'std', 'min', 'max'],
            'sentiment_confidence': ['mean', 'min', 'max'],
            'quality_score': ['mean', 'std', 'min', 'max'],
            'content_richness_score': ['mean', 'std'],
            'information_density': ['mean', 'std'],
            'total_entity_count': ['sum', 'mean', 'max'],
            'person_entity_count': ['sum', 'mean', 'max'],
            'organization_entity_count': ['sum', 'mean', 'max'],
            'location_entity_count': ['sum', 'mean', 'max']
        }
    
    def aggregate_data(self, silver_data: pd.DataFrame) -> pd.DataFrame:
        """
        Main aggregation function to create gold layer data
        """
        try:
            logger.info(f"Starting data aggregation for {len(silver_data)} silver records")
            
            if silver_data.empty:
                return pd.DataFrame()
            
            # Rename columns
            topic_agg['article_count'] = topic_agg['id_count']
            topic_agg['avg_sentiment_score'] = topic_agg['sentiment_score_mean']
            topic_agg['processing_date'] = datetime.now().date()
            
            # Select relevant columns for gold schema
            gold_columns = self._get_gold_columns()
            topic_agg = self._align_with_gold_schema(topic_agg, gold_columns)
            
            return topic_agg
            
        except Exception as e:
            logger.error(f"Error in topic aggregation: {e}")
            return pd.DataFrame()
    
    def _aggregate_by_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by sentiment"""
        try:
            sentiment_agg = df.groupby(['pub_date_only', 'sentiment_label', 'region']).agg({
                'id': 'count',
                'sentiment_score': ['mean', 'std'],
                'sentiment_confidence': 'mean',
                'quality_score': ['mean', 'std'],
                'total_entity_count': ['sum', 'mean'],
                'person_entity_count': ['sum', 'mean'],
                'organization_entity_count': ['sum', 'mean'],
                'location_entity_count': ['sum', 'mean']
            }).round(3)
            
            # Flatten column names
            sentiment_agg.columns = ['_'.join(col).strip() for col in sentiment_agg.columns]
            sentiment_agg = sentiment_agg.reset_index()
            
            # Add aggregation metadata
            sentiment_agg['aggregation_type'] = 'sentiment'
            sentiment_agg['aggregation_key'] = (
                sentiment_agg['pub_date_only'].astype(str) + '_' + 
                sentiment_agg['sentiment_label'] + '_' + 
                sentiment_agg['region']
            )
            sentiment_agg['pub_date'] = pd.to_datetime(sentiment_agg['pub_date_only'])
            sentiment_agg['source'] = 'All Sources'
            sentiment_agg['topic_name'] = 'All Topics'
            sentiment_agg['topic_id'] = 0
            sentiment_agg['media_name'] = 'All Media'
            sentiment_agg['author'] = 'Multiple Authors'
            
            # Rename columns
            sentiment_agg['article_count'] = sentiment_agg['id_count']
            sentiment_agg['avg_sentiment_score'] = sentiment_agg['sentiment_score_mean']
            sentiment_agg['processing_date'] = datetime.now().date()
            
            # Select relevant columns for gold schema
            gold_columns = self._get_gold_columns()
            sentiment_agg = self._align_with_gold_schema(sentiment_agg, gold_columns)
            
            return sentiment_agg
            
        except Exception as e:
            logger.error(f"Error in sentiment aggregation: {e}")
            return pd.DataFrame()
    
    def _aggregate_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate entity data"""
        try:
            # Get entity statistics for each date/region combination
            entity_stats = []
            
            for (date, region), group in df.groupby(['pub_date_only', 'region']):
                # Aggregate top entities
                all_persons = []
                all_orgs = []
                all_locations = []
                
                for _, row in group.iterrows():
                    if isinstance(row['person_entities'], list):
                        all_persons.extend(row['person_entities'])
                    if isinstance(row['organization_entities'], list):
                        all_orgs.extend(row['organization_entities'])
                    if isinstance(row['location_entities'], list):
                        all_locations.extend(row['location_entities'])
                
                # Get top entities
                top_persons = [item[0] for item in Counter(all_persons).most_common(5)]
                top_orgs = [item[0] for item in Counter(all_orgs).most_common(5)]
                top_locations = [item[0] for item in Counter(all_locations).most_common(5)]
                
                # Create entity aggregation record
                entity_record = {
                    'pub_date_only': date,
                    'pub_date': pd.to_datetime(date),
                    'region': region,
                    'source': 'All Sources',
                    'topic_name': 'Entity Analysis',
                    'topic_id': 999,
                    'media_name': 'All Media',
                    'sentiment_label': 'mixed',
                    'author': 'Multiple Authors',
                    'article_count': len(group),
                    'avg_sentiment_score': group['sentiment_score'].mean(),
                    'sentiment_confidence': group['sentiment_confidence'].mean(),
                    'top_persons': top_persons,
                    'top_organizations': top_orgs,
                    'top_locations': top_locations,
                    'person_entity_count': group['person_entity_count'].sum(),
                    'organization_entity_count': group['organization_entity_count'].sum(),
                    'location_entity_count': group['location_entity_count'].sum(),
                    'processing_date': datetime.now().date(),
                    'aggregation_type': 'entity',
                    'aggregation_key': f"{date}_entity_{region}"
                }
                
                entity_stats.append(entity_record)
            
            if entity_stats:
                entity_agg = pd.DataFrame(entity_stats)
                
                # Select relevant columns for gold schema
                gold_columns = self._get_gold_columns()
                entity_agg = self._align_with_gold_schema(entity_agg, gold_columns)
                
                return entity_agg
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in entity aggregation: {e}")
            return pd.DataFrame()
    
    def _combine_aggregations(self, aggregation_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine different types of aggregations"""
        try:
            valid_aggregations = [agg for agg in aggregation_list if not agg.empty]
            
            if not valid_aggregations:
                return pd.DataFrame()
            
            # Combine all aggregations
            combined_df = pd.concat(valid_aggregations, ignore_index=True)
            
            # Add sequential IDs (these will be replaced by database auto-increment)
            combined_df = combined_df.reset_index(drop=True)
            
            # Ensure all required columns exist
            gold_columns = self._get_gold_columns()
            for col in gold_columns:
                if col not in combined_df.columns:
                    if col == 'silver_id':
                        combined_df[col] = None
                    elif col in ['key_entities']:
                        combined_df[col] = combined_df.apply(self._create_key_entities, axis=1)
                    else:
                        combined_df[col] = None
            
            # Reorder columns to match gold schema
            combined_df = combined_df[gold_columns]
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error combining aggregations: {e}")
            raise
    
    def _create_key_entities(self, row) -> Dict[str, Any]:
        """Create key entities JSON from aggregated data"""
        try:
            key_entities = {
                'top_persons': row.get('top_persons', [])[:3],
                'top_organizations': row.get('top_organizations', [])[:3],
                'top_locations': row.get('top_locations', [])[:3],
                'total_person_count': int(row.get('person_entity_count', 0)),
                'total_org_count': int(row.get('organization_entity_count', 0)),
                'total_location_count': int(row.get('location_entity_count', 0))
            }
            return key_entities
        except Exception as e:
            logger.error(f"Error creating key entities: {e}")
            return {}
    
    def _get_gold_columns(self) -> List[str]:
        """Get the list of columns for gold schema"""
        return [
            'silver_id', 'pub_date', 'source', 'region', 'author',
            'topic_name', 'topic_id', 'media_name', 'sentiment_label',
            'avg_sentiment_score', 'article_count', 'sentiment_confidence',
            'key_entities', 'top_persons', 'top_organizations', 'top_locations',
            'person_entity_count', 'organization_entity_count', 'location_entity_count',
            'processing_date'
        ]
    
    def _align_with_gold_schema(self, df: pd.DataFrame, gold_columns: List[str]) -> pd.DataFrame:
        """Align dataframe with gold schema"""
        try:
            # Add missing columns with default values
            for col in gold_columns:
                if col not in df.columns:
                    if col == 'silver_id':
                        df[col] = None
                    elif col == 'key_entities':
                        df[col] = df.apply(self._create_key_entities, axis=1)
                    elif col in ['top_persons', 'top_organizations', 'top_locations']:
                        df[col] = df[col] if col in df.columns else []
                    elif col in ['person_entity_count', 'organization_entity_count', 'location_entity_count']:
                        df[col] = df.get(f'{col.split("_")[0]}_entity_count_sum', 0)
                    elif col == 'sentiment_confidence':
                        df[col] = df.get('sentiment_confidence_mean', 0.0)
                    else:
                        df[col] = df.get(col, '')
            
            # Ensure proper data types
            numeric_columns = [
                'avg_sentiment_score', 'article_count', 'sentiment_confidence',
                'person_entity_count', 'organization_entity_count', 'location_entity_count'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Ensure date columns are proper datetime/date
            if 'pub_date' in df.columns:
                df['pub_date'] = pd.to_datetime(df['pub_date'])
            if 'processing_date' in df.columns:
                df['processing_date'] = pd.to_datetime(df['processing_date']).dt.date
            
            return df[gold_columns]
            
        except Exception as e:
            logger.error(f"Error aligning with gold schema: {e}")
            return df
    
    def create_summary_aggregations(self, df: pd.DataFrame, period: str = 'daily') -> Dict[str, Any]:
        """Create summary aggregations for reporting"""
        try:
            if df.empty:
                return {}
            
            summary = {
                'total_articles': int(df['article_count'].sum()),
                'avg_sentiment_score': float(df['avg_sentiment_score'].mean()),
                'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
                'top_sources': df.groupby('source')['article_count'].sum().nlargest(10).to_dict(),
                'top_regions': df.groupby('region')['article_count'].sum().nlargest(10).to_dict(),
                'date_range': {
                    'start': df['pub_date'].min().isoformat() if not df['pub_date'].isna().all() else None,
                    'end': df['pub_date'].max().isoformat() if not df['pub_date'].isna().all() else None
                },
                'entity_stats': {
                    'total_persons': int(df['person_entity_count'].sum()),
                    'total_organizations': int(df['organization_entity_count'].sum()),
                    'total_locations': int(df['location_entity_count'].sum())
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating summary aggregations: {e}")
            return {} Prepare data for aggregation
            prepared_data = self._prepare_data_for_aggregation(silver_data)
            
            # Perform different types of aggregations
            daily_aggregations = self._aggregate_by_date(prepared_data)
            source_aggregations = self._aggregate_by_source(prepared_data)
            topic_aggregations = self._aggregate_by_topic(prepared_data)
            sentiment_aggregations = self._aggregate_by_sentiment(prepared_data)
            entity_aggregations = self._aggregate_entities(prepared_data)
            
            # Combine all aggregations
            combined_aggregations = self._combine_aggregations([
                daily_aggregations,
                source_aggregations,
                topic_aggregations,
                sentiment_aggregations,
                entity_aggregations
            ])
            
            logger.info(f"Data aggregation completed. Generated {len(combined_aggregations)} aggregated records")
            return combined_aggregations
            
        except Exception as e:
            logger.error(f"Error in data aggregation: {e}")
            raise
    
    def _prepare_data_for_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for aggregation"""
        try:
            prepared_df = df.copy()
            
            # Ensure pub_date is datetime
            prepared_df['pub_date'] = pd.to_datetime(prepared_df['pub_date'])
            
            # Create date columns for aggregation
            prepared_df['pub_date_only'] = prepared_df['pub_date'].dt.date
            prepared_df['pub_year'] = prepared_df['pub_date'].dt.year
            prepared_df['pub_month'] = prepared_df['pub_date'].dt.month
            prepared_df['pub_week'] = prepared_df['pub_date'].dt.isocalendar().week
            prepared_df['pub_day_of_week'] = prepared_df['pub_date'].dt.dayofweek
            
            # Clean and standardize categorical fields
            prepared_df['source'] = prepared_df['source'].fillna('Unknown').astype(str)
            prepared_df['region'] = prepared_df['region'].fillna('Unknown').astype(str)
            prepared_df['sentiment_label'] = prepared_df['sentiment_label'].fillna('neutral').astype(str)
            prepared_df['content_type'] = prepared_df['content_type'].fillna('news').astype(str)
            
            # Handle numeric fields
            numeric_fields = [
                'sentiment_score', 'sentiment_confidence', 'quality_score',
                'content_richness_score', 'information_density',
                'total_entity_count', 'person_entity_count',
                'organization_entity_count', 'location_entity_count'
            ]
            
            for field in numeric_fields:
                if field in prepared_df.columns:
                    prepared_df[field] = pd.to_numeric(prepared_df[field], errors='coerce').fillna(0)
            
            # Extract top entities for aggregation
            prepared_df = self._extract_top_entities(prepared_df)
            
            return prepared_df
            
        except Exception as e:
            logger.error(f"Error preparing data for aggregation: {e}")
            raise
    
    def _extract_top_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract top entities for each record"""
        try:
            # Extract top 3 entities of each type
            df['top_persons'] = df['person_entities'].apply(
                lambda x: x[:3] if isinstance(x, list) else []
            )
            df['top_organizations'] = df['organization_entities'].apply(
                lambda x: x[:3] if isinstance(x, list) else []
            )
            df['top_locations'] = df['location_entities'].apply(
                lambda x: x[:3] if isinstance(x, list) else []
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting top entities: {e}")
            return df
    
    def _aggregate_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by date"""
        try:
            # Daily aggregation
            daily_agg = df.groupby(['pub_date_only']).agg({
                'id': 'count',
                'sentiment_score': ['mean', 'std'],
                'sentiment_confidence': 'mean',
                'quality_score': ['mean', 'std'],
                'total_entity_count': ['sum', 'mean'],
                'person_entity_count': ['sum', 'mean'],
                'organization_entity_count': ['sum', 'mean'],
                'location_entity_count': ['sum', 'mean']
            }).round(3)
            
            # Flatten column names
            daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns]
            daily_agg = daily_agg.reset_index()
            
            # Add aggregation metadata
            daily_agg['aggregation_type'] = 'daily'
            daily_agg['aggregation_key'] = daily_agg['pub_date_only'].astype(str)
            daily_agg['pub_date'] = pd.to_datetime(daily_agg['pub_date_only'])
            daily_agg['source'] = 'All Sources'
            daily_agg['region'] = 'All Regions'
            daily_agg['topic_name'] = 'All Topics'
            daily_agg['topic_id'] = 0
            daily_agg['media_name'] = 'All Media'
            daily_agg['sentiment_label'] = 'mixed'
            daily_agg['author'] = 'Multiple Authors'
            
            # Rename count column
            daily_agg['article_count'] = daily_agg['id_count']
            daily_agg['avg_sentiment_score'] = daily_agg['sentiment_score_mean']
            daily_agg['processing_date'] = datetime.now().date()
            
            # Select relevant columns for gold schema
            gold_columns = self._get_gold_columns()
            daily_agg = self._align_with_gold_schema(daily_agg, gold_columns)
            
            return daily_agg
            
        except Exception as e:
            logger.error(f"Error in date aggregation: {e}")
            return pd.DataFrame()
    
    def _aggregate_by_source(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by source and date"""
        try:
            source_agg = df.groupby(['pub_date_only', 'source', 'region']).agg({
                'id': 'count',
                'sentiment_score': ['mean', 'std'],
                'sentiment_confidence': 'mean',
                'quality_score': ['mean', 'std'],
                'total_entity_count': ['sum', 'mean'],
                'person_entity_count': ['sum', 'mean'],
                'organization_entity_count': ['sum', 'mean'],
                'location_entity_count': ['sum', 'mean']
            }).round(3)
            
            # Flatten column names
            source_agg.columns = ['_'.join(col).strip() for col in source_agg.columns]
            source_agg = source_agg.reset_index()
            
            # Add aggregation metadata
            source_agg['aggregation_type'] = 'source'
            source_agg['aggregation_key'] = (
                source_agg['pub_date_only'].astype(str) + '_' + 
                source_agg['source'] + '_' + 
                source_agg['region']
            )
            source_agg['pub_date'] = pd.to_datetime(source_agg['pub_date_only'])
            source_agg['topic_name'] = 'All Topics'
            source_agg['topic_id'] = 0
            source_agg['media_name'] = source_agg['source']
            source_agg['sentiment_label'] = 'mixed'
            source_agg['author'] = 'Multiple Authors'
            
            # Rename columns
            source_agg['article_count'] = source_agg['id_count']
            source_agg['avg_sentiment_score'] = source_agg['sentiment_score_mean']
            source_agg['processing_date'] = datetime.now().date()
            
            # Select relevant columns for gold schema
            gold_columns = self._get_gold_columns()
            source_agg = self._align_with_gold_schema(source_agg, gold_columns)
            
            return source_agg
            
        except Exception as e:
            logger.error(f"Error in source aggregation: {e}")
            return pd.DataFrame()
    
    def _aggregate_by_topic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by topic/content type"""
        try:
            topic_agg = df.groupby(['pub_date_only', 'content_type', 'region']).agg({
                'id': 'count',
                'sentiment_score': ['mean', 'std'],
                'sentiment_confidence': 'mean',
                'quality_score': ['mean', 'std'],
                'total_entity_count': ['sum', 'mean'],
                'person_entity_count': ['sum', 'mean'],
                'organization_entity_count': ['sum', 'mean'],
                'location_entity_count': ['sum', 'mean']
            }).round(3)
            
            # Flatten column names
            topic_agg.columns = ['_'.join(col).strip() for col in topic_agg.columns]
            topic_agg = topic_agg.reset_index()
            
            # Add aggregation metadata
            topic_agg['aggregation_type'] = 'topic'
            topic_agg['aggregation_key'] = (
                topic_agg['pub_date_only'].astype(str) + '_' + 
                topic_agg['content_type'] + '_' + 
                topic_agg['region']
            )
            topic_agg['pub_date'] = pd.to_datetime(topic_agg['pub_date_only'])
            topic_agg['source'] = 'All Sources'
            topic_agg['topic_name'] = topic_agg['content_type']
            topic_agg['topic_id'] = topic_agg['content_type'].astype('category').cat.codes + 1
            topic_agg['media_name'] = 'All Media'
            topic_agg['sentiment_label'] = 'mixed'
            topic_agg['author'] = 'Multiple Authors'
            
            #