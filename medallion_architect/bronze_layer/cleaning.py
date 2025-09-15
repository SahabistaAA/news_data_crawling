import logging
import pandas as pd
import re
from typing import Dict, Any
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning operations for bronze layer"""
    
    def __init__(self):
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning function that orchestrates all cleaning operations
        """
        try:
            logger.info("Starting data cleaning process")
            
            # Make a copy to avoid modifying the original DataFrame
            cleaned_df = df.copy()
            
            # Apply cleaning steps in sequence
            cleaned_df = self._handle_missing_values(cleaned_df)
            cleaned_df = self._clean_text_fields(cleaned_df)
            cleaned_df = self._remove_duplicates(cleaned_df)
            cleaned_df = self._validate_essential_fields(cleaned_df)
            cleaned_df = self._add_quality_metrics(cleaned_df)
            
            logger.info(f"Data cleaning completed. {len(cleaned_df)} records processed")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {e}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate defaults"""
        try:
            # Text fields - fill with empty string
            text_columns = ['title', 'author', 'content', 'full_content', 'source', 'region']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('')
            
            # URL field - fill with empty string
            if 'url' in df.columns:
                df['url'] = df['url'].fillna('')
            
            # Date field - fill with current timestamp if missing
            if 'pub_date' in df.columns:
                df['pub_date'] = df['pub_date'].fillna(pd.Timestamp.now())
            
            logger.info("Missing values handled")
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and sanitize text fields"""
        try:
            text_columns = ['title', 'content', 'full_content', 'author']
            
            for col in text_columns:
                if col in df.columns:
                    # Convert to string and strip whitespace
                    df[col] = df[col].astype(str).str.strip()
                    
                    # Remove HTML tags
                    df[col] = df[col].apply(self._remove_html_tags)
                    
                    # Clean excessive whitespace
                    df[col] = df[col].apply(self._clean_whitespace)
                    
                    # Remove control characters
                    df[col] = df[col].apply(self._remove_control_chars)
            
            # Special handling for URL field
            if 'url' in df.columns:
                df['url'] = df['url'].astype(str).str.strip()
                df['url'] = df['url'].apply(self._validate_url)
            
            logger.info("Text fields cleaned")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning text fields: {e}")
            raise
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return self.html_pattern.sub('', text)
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean excessive whitespace"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        # Replace multiple whitespace with single space
        return re.sub(r'\s+', ' ', text).strip()
    
    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        # Remove control characters but keep newlines and tabs
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    def _validate_url(self, url: str) -> str:
        """Basic URL validation and cleaning"""
        if pd.isna(url) or not isinstance(url, str):
            return ""
        
        url = url.strip()
        
        # Add protocol if missing
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Basic URL format validation
        if not self.url_pattern.match(url):
            return ""
        
        return url
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates based on URL and content similarity"""
        try:
            initial_count = len(df)
            
            # Remove exact URL duplicates (keep first occurrence)
            if 'url' in df.columns:
                df = df.drop_duplicates(subset=['url'], keep='first')
            
            # Remove duplicates based on title and content similarity
            df = self._remove_content_duplicates(df)
            
            removed_count = initial_count - len(df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            raise
    
    def _remove_content_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates based on content hash"""
        try:
            # Create content hash for duplicate detection
            df['content_hash'] = df.apply(
                lambda row: self._generate_content_hash(
                    str(row.get('title', '')), 
                    str(row.get('content', ''))
                ), 
                axis=1
            )
            
            # Remove duplicates based on content hash
            df = df.drop_duplicates(subset=['content_hash'], keep='first')
            
            # Mark remaining records for duplicate status
            df['is_duplicate'] = False
            
            # Drop the temporary hash column
            df = df.drop(columns=['content_hash'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing content duplicates: {e}")
            raise
    
    def _generate_content_hash(self, title: str, content: str) -> str:
        """Generate hash for content deduplication"""
        try:
            combined_content = f"{title.lower().strip()}{content.lower().strip()[:500]}"
            return hashlib.md5(combined_content.encode()).hexdigest()
        except Exception:
            return ""
    
    def _validate_essential_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and filter records with missing essential fields"""
        try:
            initial_count = len(df)
            
            # Essential fields that cannot be empty
            essential_fields = ['title', 'content']
            
            for field in essential_fields:
                if field in df.columns:
                    # Filter out records where essential fields are empty
                    mask = (df[field].astype(str).str.len() > 0) & (df[field] != 'nan')
                    df = df[mask]
            
            # Additional validation for URL format
            if 'url' in df.columns:
                # Keep records with either valid URL or empty URL (for cases where URL is not available)
                valid_url_mask = (df['url'] == '') | df['url'].str.match(r'https?://.+')
                df = df[valid_url_mask]
            
            removed_count = initial_count - len(df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} records with invalid essential fields")
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating essential fields: {e}")
            raise
    
    def _add_quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality check metrics to the data"""
        try:
            df['quality_check_status'] = df.apply(self._assess_record_quality, axis=1)
            df['cleaned_at'] = pd.Timestamp.now()
            
            quality_distribution = df['quality_check_status'].value_counts()
            logger.info(f"Quality distribution: {quality_distribution.to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding quality metrics: {e}")
            raise
    
    def _assess_record_quality(self, row) -> str:
        """Assess the quality of a single record"""
        try:
            score = 0
            max_score = 5
            
            # Check title quality
            title_len = len(str(row.get('title', '')))
            if title_len > 10:
                score += 1
            
            # Check content quality
            content_len = len(str(row.get('content', '')))
            if content_len > 100:
                score += 1
            if content_len > 500:
                score += 1
            
            # Check if URL is present and valid
            url = str(row.get('url', ''))
            if url and url.startswith(('http://', 'https://')):
                score += 1
            
            # Check if author is present
            author = str(row.get('author', ''))
            if author and len(author) > 2:
                score += 1
            
            # Determine quality level
            if score >= 4:
                return 'high'
            elif score >= 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing record quality: {e}")
            return 'unknown'