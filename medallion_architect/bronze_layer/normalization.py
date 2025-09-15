import logging
import pandas as pd
from typing import Dict, List, Optional
import re
from datetime import datetime, timezone
import pytz
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Handles data normalization operations for bronze layer"""
    
    def __init__(self):
        # Timezone mappings for different regions
        self.timezone_map = {
            'indonesia': 'Asia/Jakarta',
            'singapore': 'Asia/Singapore',
            'malaysia': 'Asia/Kuala_Lumpur',
            'thailand': 'Asia/Bangkok',
            'philippines': 'Asia/Manila',
            'vietnam': 'Asia/Ho_Chi_Minh',
            'global': 'UTC'
        }
        
        # Source domain mappings
        self.source_domains = {}
        
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main normalization function
        """
        try:
            logger.info("Starting data normalization")
            
            normalized_df = df.copy()
            
            # Apply normalization steps
            normalized_df = self._normalize_datetime_fields(normalized_df)
            normalized_df = self._normalize_text_case(normalized_df)
            normalized_df = self._normalize_source_names(normalized_df)
            normalized_df = self._normalize_regions(normalized_df)
            normalized_df = self._normalize_author_names(normalized_df)
            normalized_df = self._extract_domain_info(normalized_df)
            
            logger.info("Data normalization completed")
            return normalized_df
            
        except Exception as e:
            logger.error(f"Error in data normalization: {e}")
            raise
    
    def _normalize_datetime_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize datetime fields to consistent format and timezone"""
        try:
            if 'pub_date' not in df.columns:
                return df
            
            # Convert to datetime if not already
            df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
            
            # Handle timezone normalization
            df['pub_date_normalized'] = df.apply(self._normalize_single_datetime, axis=1)
            
            # Keep original pub_date and add normalized version
            df['pub_date'] = df['pub_date_normalized']
            df = df.drop(columns=['pub_date_normalized'])
            
            # Ensure ingested_at is set
            if 'ingested_at' not in df.columns:
                df['ingested_at'] = pd.Timestamp.now(tz=timezone.utc)
            else:
                df['ingested_at'] = pd.to_datetime(df['ingested_at'], errors='coerce')
                df['ingested_at'] = df['ingested_at'].fillna(pd.Timestamp.now(tz=timezone.utc))
            
            logger.info("DateTime fields normalized")
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing datetime fields: {e}")
            raise
    
    def _normalize_single_datetime(self, row) -> pd.Timestamp:
        """Normalize a single datetime value based on region"""
        try:
            pub_date = row.get('pub_date')
            region = str(row.get('region', 'global')).lower()
            
            if pd.isna(pub_date):
                return pd.Timestamp.now(tz=timezone.utc)
            
            # If datetime is timezone-naive, localize it based on region
            if pub_date.tz is None:
                region_tz = self.timezone_map.get(region, 'UTC')
                try:
                    local_tz = pytz.timezone(region_tz)
                    localized_dt = local_tz.localize(pub_date)
                    # Convert to UTC
                    return localized_dt.astimezone(pytz.UTC)
                except Exception:
                    # Fallback to UTC
                    return pub_date.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC if already timezone-aware
                return pub_date.astimezone(pytz.UTC)
                
        except Exception as e:
            logger.error(f"Error normalizing single datetime: {e}")
            return pd.Timestamp.now(tz=timezone.utc)
    
    def _normalize_text_case(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text case for consistency"""
        try:
            # Title case for titles
            if 'title' in df.columns:
                df['title'] = df['title'].astype(str).apply(self._normalize_title_case)
            
            # Proper case for author names
            if 'author' in df.columns:
                df['author'] = df['author'].astype(str).apply(self._normalize_author_case)
            
            # Lowercase for regions and sources (for consistency in grouping)
            if 'region' in df.columns:
                df['region'] = df['region'].astype(str).str.lower().str.strip()
            
            if 'source' in df.columns:
                df['source'] = df['source'].astype(str).str.strip()
            
            logger.info("Text case normalized")
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing text case: {e}")
            raise
    
    def _normalize_title_case(self, title: str) -> str:
        """Normalize title to proper title case"""
        try:
            if not title or title == 'nan':
                return ""
            
            # Convert to title case, but handle common abbreviations
            title = title.strip()
            
            # Handle all caps titles
            if title.isupper():
                title = title.lower()
            
            # Apply title case
            title = title.title()
            
            # Fix common words that shouldn't be capitalized
            common_words = ['And', 'Or', 'But', 'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By']
            for word in common_words:
                title = re.sub(rf'\b{word}\b', word.lower(), title)
            
            # Ensure first word is always capitalized
            if title:
                title = title[0].upper() + title[1:]
            
            return title
            
        except Exception as e:
            logger.error(f"Error normalizing title case: {e}")
            return str(title)
    
    def _normalize_author_case(self, author: str) -> str:
        """Normalize author names to proper case"""
        try:
            if not author or author == 'nan':
                return ""
            
            author = author.strip()
            
            # Handle multiple authors separated by common delimiters
            delimiters = [',', ' and ', ' & ', ';']
            authors = [author]
            
            for delimiter in delimiters:
                temp_authors = []
                for auth in authors:
                    temp_authors.extend(auth.split(delimiter))
                authors = temp_authors
            
            # Normalize each author name
            normalized_authors = []
            for auth in authors:
                auth = auth.strip()
                if auth:
                    # Convert to title case
                    auth = ' '.join(word.capitalize() for word in auth.split())
                    normalized_authors.append(auth)
            
            return ', '.join(normalized_authors)
            
        except Exception as e:
            logger.error(f"Error normalizing author case: {e}")
            return str(author)
    
    def _normalize_source_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize source names for consistency"""
        try:
            if 'source' not in df.columns:
                return df
            
            # Create mapping of common source variations
            source_mapping = {
                'cnn': 'CNN',
                'bbc': 'BBC',
                'reuters': 'Reuters',
                'ap news': 'Associated Press',
                'associated press': 'Associated Press',
                'the new york times': 'The New York Times',
                'nyt': 'The New York Times',
                'the guardian': 'The Guardian',
                'guardian': 'The Guardian',
                'washington post': 'The Washington Post',
                'wapo': 'The Washington Post',
                'kompas': 'Kompas',
                'detik': 'Detik',
                'tempo': 'Tempo',
                'liputan6': 'Liputan6'
            }
            
            df['source'] = df['source'].astype(str).apply(
                lambda x: self._map_source_name(x, source_mapping)
            )
            
            logger.info("Source names normalized")
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing source names: {e}")
            raise
    
    def _map_source_name(self, source: str, mapping: Dict[str, str]) -> str:
        """Map source name using the provided mapping"""
        try:
            if not source or source == 'nan':
                return ""
            
            source_lower = source.lower().strip()
            
            # Check for exact matches first
            if source_lower in mapping:
                return mapping[source_lower]
            
            # Check for partial matches
            for key, value in mapping.items():
                if key in source_lower:
                    return value
            
            # If no mapping found, return title case version
            return source.strip().title()
            
        except Exception as e:
            logger.error(f"Error mapping source name: {e}")
            return str(source)
    
    def _normalize_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize region names for consistency"""
        try:
            if 'region' not in df.columns:
                return df
            
            # Create mapping of region variations
            region_mapping = {
                'id': 'indonesia',
                'idn': 'indonesia',
                'indonesia': 'indonesia',
                'sg': 'singapore',
                'sgp': 'singapore',
                'singapore': 'singapore',
                'my': 'malaysia',
                'mys': 'malaysia',
                'malaysia': 'malaysia',
                'th': 'thailand',
                'tha': 'thailand',
                'thailand': 'thailand',
                'ph': 'philippines',
                'phl': 'philippines',
                'philippines': 'philippines',
                'vn': 'vietnam',
                'vnm': 'vietnam',
                'vietnam': 'vietnam',
                'global': 'global',
                'international': 'global',
                'worldwide': 'global'
            }
            
            df['region'] = df['region'].astype(str).str.lower().str.strip()
            df['region'] = df['region'].map(region_mapping).fillna(df['region'])
            
            logger.info("Regions normalized")
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing regions: {e}")
            raise
    
    def _normalize_author_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Additional author name normalization"""
        try:
            if 'author' not in df.columns:
                return df
            
            # Remove common prefixes and suffixes
            df['author'] = df['author'].astype(str).apply(self._clean_author_name)
            
            logger.info("Author names normalized")
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing author names: {e}")
            raise
    
    def _clean_author_name(self, author: str) -> str:
        """Clean author name by removing common prefixes/suffixes"""
        try:
            if not author or author == 'nan':
                return ""
            
            author = author.strip()
            
            # Remove common prefixes
            prefixes_to_remove = ['by ', 'written by ', 'author: ', 'reporter: ']
            for prefix in prefixes_to_remove:
                if author.lower().startswith(prefix):
                    author = author[len(prefix):].strip()
            
            # Remove common suffixes
            suffixes_to_remove = [' reporter', ' correspondent', ' journalist']
            for suffix in suffixes_to_remove:
                if author.lower().endswith(suffix):
                    author = author[:-len(suffix)].strip()
            
            return author
            
        except Exception as e:
            logger.error(f"Error cleaning author name: {e}")
            return str(author)
    
    def _extract_domain_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and normalize domain information from URLs"""
        try:
            if 'url' not in df.columns:
                return df
            
            df['domain'] = df['url'].astype(str).apply(self._extract_domain)
            
            logger.info("Domain information extracted")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting domain info: {e}")
            raise
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            if not url or url == 'nan':
                return ""
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
            
        except Exception as e:
            logger.error(f"Error extracting domain: {e}")
            return ""