import logging
import pandas as pd
from typing import List, Dict, Any
from .cleaning import DataCleaner
from .normalization import DataNormalizer
from .standardization import DataStandardizer
from ..database.bronze_db import BronzeDB

logger = logging.getLogger(__name__)


class BronzeProcessor:
    """Main processor for bronze layer operations"""
    
    def __init__(self):
        self.cleaner = DataCleaner()
        self.normalizer = DataNormalizer()
        self.standardizer = DataStandardizer()
        self.db = BronzeDB()
    
    def process_raw_to_bronze(self) -> None:
        """
        Main pipeline to process raw data into bronze layer
        """
        try:
            logger.info("Starting bronze layer processing")
            
            # Create bronze table if it doesn't exist
            self.db.create_bronze_table()
            
            # Fetch raw data
            raw_data = self.db.fetch_raw_data()
            
            if raw_data.empty:
                logger.info("No raw data found to process")
                return
            
            logger.info(f"Processing {len(raw_data)} raw records")
            
            # Process through the pipeline
            cleaned_data = self.cleaner.clean_data(raw_data)
            normalized_data = self.normalizer.normalize_data(cleaned_data)
            standardized_data = self.standardizer.standardize_data(normalized_data)
            
            # Save to bronze table
            self.db.save_to_bronze(standardized_data)
            
            logger.info("Bronze layer processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in bronze layer processing: {e}")
            raise
    
    def get_bronze_data(self, limit: int = None) -> pd.DataFrame:
        """
        Retrieve bronze data for silver layer processing
        """
        try:
            return self.db.fetch_bronze_data(limit=limit)
        except Exception as e:
            logger.error(f"Error fetching bronze data: {e}")
            raise
    
    def mark_as_processed(self, record_ids: List[int]) -> None:
        """
        Mark bronze records as processed
        """
        try:
            self.db.mark_as_processed(record_ids)
        except Exception as e:
            logger.error(f"Error marking records as processed: {e}")
            raise