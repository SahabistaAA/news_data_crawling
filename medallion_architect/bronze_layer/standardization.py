import logging
import pandas as pd
from typing import Dict, List, Any, Optional
import re
import unicodedata
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class DataStandardizer:
    """Handles data standardization operations for bronze layer"""
    
    def __init__(self):
        # Language mappings
        self.language_map = {
            'id': 'indonesian',
            'en': 'english',
            'ms': 'malay',
            'th': 'thai',
            'vi': 'vietnamese',
            'zh': 'chinese',
            'ja': 'japanese',
            'ko': 'korean'
        }
        
        # Content type patterns
        self.content_patterns = {
            'news': r'(news|berita|artikel|story)',
            'opinion': r'(opinion|opini|editorial|comment)',
            'interview': r'(interview|wawancara|tanya|jawab)',
            'analysis': r'(analysis|analisis|review|ulasan)',
            'feature': r'(feature|laporan|khusus|special)'
        }
    
    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main standardization function
        """
        try:
            logger.info("Starting data standardization")
            
            standardized_df = df.copy()
            
            # Apply standardization steps
            standardized_df = self._standardize_text_encoding(standardized_df)
            standardized_df = self._detect_language(standardized_df)
            standardized_df = self._standardize_content_length(standardized_df)
            standardized_df = self._classify_content_type(standardized_df)
            standardized_df = self._add_metadata(standardized_df)
            standardized_df = self._validate_final_schema(standardized_df)
            
            logger.info("Data standardization completed")
            return standardized_df
            
        except Exception as e:
            logger.error(f"Error in data standardization: {e}")
            raise
    
    def _standardize_text_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text encoding to UTF-8"""
        try:
            text_columns = ['title', 'content', 'full_content', 'author']
            
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).apply(self._normalize_unicode)
            
            logger.info("Text encoding standardized")
            return df
            
        except Exception as e:
            logger.error(f"Error standardizing text encoding: {e}")
            raise
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text"""
        try:
            if not text or text == 'nan':
                return ""
            
            # Normalize to NFC (Canonical Decomposition, followed by Canonical Composition)
            text = unicodedata.normalize('NFC', text)
            
            # Remove or replace problematic characters
            text = text.replace('\u200b', '')  # Zero-width space
            text = text.replace('\u200c', '')  # Zero-width non-joiner
            text = text.replace('\u200d', '')  # Zero-width joiner
            text = text.replace('\ufeff', '')  # Byte order mark
            
            # Replace smart quotes with regular quotes
            text = text.replace('\u2018', "'").replace('\u2019', "'")  # Smart single quotes
            text = text.replace('\u201c', '"').replace('\u201d', '"')  # Smart double quotes
            
            # Replace em dash and en dash with regular dash
            text = text.replace('\u2013', '-').replace('\u2014', '-')
            
            return text
            
        except Exception as e:
            logger.error(f"Error normalizing unicode: {e}")
            return str(text)
    
    def _detect_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect language of content"""
        try:
            df['language_detected'] = df.apply(self._detect_single_language, axis=1)
            
            # Log language distribution
            lang_distribution = df['language_detected'].value_counts()
            logger.info(f"Language distribution: {lang_distribution.to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            raise
    
    def _detect_single_language(self, row) -> str:
        """Detect language for a single record"""
        try:
            # Combine title and content for better detection
            title = str(row.get('title', ''))
            content = str(row.get('content', ''))
            
            text_to_detect = f"{title} {content}".strip()
            
            # Skip if text is too short
            if len(text_to_detect) < 20:
                return 'unknown'
            
            # Use only first 1000 characters for performance
            text_sample = text_to_detect[:1000]
            
            try:
                detected_lang = detect(text_sample)
                return self.language_map.get(detected_lang, detected_lang)
            except LangDetectException:
                # Fallback: try to detect based on region
                region = str(row.get('region', '')).lower()
                if region in ['indonesia', 'id']:
                    return 'indonesian'
                elif region in ['singapore', 'malaysia']:
                    return 'english'  # Default for these regions
                else:
                    return 'english'  # Default fallback
                    
        except Exception as e:
            logger.error(f"Error detecting single language: {e}")
            return 'unknown'
    
    def _standardize_content_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize and validate content length"""
        try:
            # Add content length metrics
            df['title_length'] = df['title'].astype(str).str.len()
            df['content_length'] = df['content'].astype(str).str.len()
            
            if 'full_content' in df.columns:
                df['full_content_length'] = df['full_content'].astype(str).str.len()
            
            # Truncate extremely long content to prevent processing issues
            max_content_length = 50000  # 50k characters
            max_title_length = 500     # 500 characters
            
            df['content'] = df['content'].astype(str).apply(
                lambda x: x[:max_content_length] if len(x) > max_content_length else x
            )
            
            df['title'] = df['title'].astype(str).apply(
                lambda x: x[:max_title_length] if len(x) > max_title_length else x
            )
            
            # Flag truncated content
            df['content_truncated'] = df['content_length'] > max_content_length
            df['title_truncated'] = df['title_length'] > max_title_length
            
            logger.info("Content length standardized")
            return df
            
        except Exception as e:
            logger.error(f"Error standardizing content length: {e}")
            raise
    
    def _classify_content_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify content type based on title and content patterns"""
        try:
            df['content_type'] = df.apply(self._detect_content_type, axis=1)
            
            # Log content type distribution
            type_distribution = df['content_type'].value_counts()
            logger.info(f"Content type distribution: {type_distribution.to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error classifying content type: {e}")
            raise
    
    def _detect_content_type(self, row) -> str:
        """Detect content type for a single record"""
        try:
            title = str(row.get('title', '')).lower()
            content = str(row.get('content', ''))[:500].lower()  # First 500 chars
            
            combined_text = f"{title} {content}"
            
            # Check patterns in order of specificity
            for content_type, pattern in self.content_patterns.items():
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return content_type
            
            # Default classification
            return 'news'
            
        except Exception as e:
            logger.error(f"Error detecting content type: {e}")
            return 'unknown'
    
    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standardized metadata fields"""
        try:
            # Add processing timestamps
            current_time = pd.Timestamp.now()
            
            if 'cleaned_at' not in df.columns:
                df['cleaned_at'] = current_time
            
            # Add word counts
            df['title_word_count'] = df['title'].astype(str).apply(
                lambda x: len(x.split()) if x and x != 'nan' else 0
            )
            
            df['content_word_count'] = df['content'].astype(str).apply(
                lambda x: len(x.split()) if x and x != 'nan' else 0
            )
            
            # Add readability score (simple metric)
            df['readability_score'] = df.apply(self._calculate_readability, axis=1)
            
            # Add quality score
            df['quality_score'] = df.apply(self._calculate_quality_score, axis=1)
            
            logger.info("Metadata added")
            return df
            
        except Exception as e:
            logger.error(f"Error adding metadata: {e}")
            raise
    
    def _calculate_readability(self, row) -> float:
        """Calculate simple readability score"""
        try:
            content = str(row.get('content', ''))
            if not content or content == 'nan':
                return 0.0
            
            words = content.split()
            if len(words) == 0:
                return 0.0
            
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) == 0:
                return 0.0
            
            # Simple readability metric: average words per sentence
            avg_words_per_sentence = len(words) / len(sentences)
            
            # Normalize to 0-1 scale (assuming 15 words per sentence as ideal)
            score = max(0, min(1, 1 - abs(avg_words_per_sentence - 15) / 15))
            
            return round(score, 3)
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.0
    
    def _calculate_quality_score(self, row) -> float:
        """Calculate overall quality score"""
        try:
            score = 0.0
            max_score = 1.0
            
            # Title quality (0.2 weight)
            title_len = len(str(row.get('title', '')))
            if 10 <= title_len <= 100:
                score += 0.2
            elif title_len > 0:
                score += 0.1
            
            # Content quality (0.3 weight)
            content_len = len(str(row.get('content', '')))
            if content_len > 1000:
                score += 0.3
            elif content_len > 300:
                score += 0.2
            elif content_len > 100:
                score += 0.1
            
            # Author presence (0.1 weight)
            author = str(row.get('author', ''))
            if author and author != 'nan' and len(author) > 2:
                score += 0.1
            
            # URL presence (0.1 weight)
            url = str(row.get('url', ''))
            if url and url.startswith(('http://', 'https://')):
                score += 0.1
            
            # Language detection (0.1 weight)
            lang = str(row.get('language_detected', ''))
            if lang and lang != 'unknown':
                score += 0.1
            
            # Date presence (0.1 weight)
            pub_date = row.get('pub_date')
            if pd.notna(pub_date):
                score += 0.1
            
            # Source presence (0.1 weight)
            source = str(row.get('source', ''))
            if source and source != 'nan' and len(source) > 0:
                score += 0.1
            
            return round(min(score, max_score), 3)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def _validate_final_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate final schema and ensure all required columns exist"""
        try:
            required_columns = [
                'title', 'url', 'pub_date', 'source', 'region', 'author',
                'content', 'full_content', 'ingested_at', 'cleaned_at',
                'is_duplicate', 'quality_check_status'
            ]
            
            # Add missing columns with default values
            for col in required_columns:
                if col not in df.columns:
                    if col in ['pub_date', 'ingested_at', 'cleaned_at']:
                        df[col] = pd.Timestamp.now()
                    elif col == 'is_duplicate':
                        df[col] = False
                    elif col == 'quality_check_status':
                        df[col] = 'unknown'
                    else:
                        df[col] = ''
            
            # Ensure data types are correct
            df['is_duplicate'] = df['is_duplicate'].astype(bool)
            df['pub_date'] = pd.to_datetime(df['pub_date'])
            df['ingested_at'] = pd.to_datetime(df['ingested_at'])
            df['cleaned_at'] = pd.to_datetime(df['cleaned_at'])
            
            # Reorder columns to match schema
            column_order = required_columns + [col for col in df.columns if col not in required_columns]
            df = df.reindex(columns=column_order)
            
            logger.info("Final schema validated")
            return df
            
        except Exception as e:
            logger.error(f"Error validating final schema: {e}")
            raise