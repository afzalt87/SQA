import pandas as pd
from typing import List, Tuple, Optional
from service.utils.config import get_logger

logger = get_logger()

class SensitiveTermsCSVValidator:
    """Validates and processes override CSV files for sensitive terms checking."""
    
    REQUIRED_COLUMNS = ['query']
    MAX_QUERY_LENGTH = 200
    
    @staticmethod
    def validate_and_load(filepath: str) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """
        Validate and load CSV file for sensitive terms checking.
        
        Returns:
            Tuple of (is_valid, dataframe, error_message)
        """
        try:
            # Load CSV
            df = pd.read_csv(filepath, dtype=str)
            
            # Check required columns
            missing_cols = set(SensitiveTermsCSVValidator.REQUIRED_COLUMNS) - set(df.columns)
            if missing_cols:
                return False, None, f"Missing required columns: {missing_cols}"
            
            # Remove empty queries
            df = df.dropna(subset=['query'])
            df = df[df['query'].str.strip() != '']
            
            if len(df) == 0:
                return False, None, "No valid queries found in CSV"
            
            # Validate query lengths
            long_queries = df[df['query'].str.len() > SensitiveTermsCSVValidator.MAX_QUERY_LENGTH]
            if len(long_queries) > 0:
                logger.warning(f"Found {len(long_queries)} queries exceeding max length")
                df = df[df['query'].str.len() <= SensitiveTermsCSVValidator.MAX_QUERY_LENGTH]
            
            # Clean queries
            df['query'] = df['query'].str.strip()
            
            # Remove duplicates
            original_count = len(df)
            df = df.drop_duplicates(subset=['query'])
            if len(df) < original_count:
                logger.info(f"Removed {original_count - len(df)} duplicate queries")
            
            logger.info(f"Successfully validated CSV with {len(df)} queries")
            return True, df, ""
            
        except FileNotFoundError:
            return False, None, f"File not found: {filepath}"
        except pd.errors.EmptyDataError:
            return False, None, "CSV file is empty"
        except Exception as e:
            return False, None, f"Error reading CSV: {str(e)}"