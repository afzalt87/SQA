from typing import Dict, List, Tuple, Any, Union
from service.utils.config import get_logger, get_env_settings

logger = get_logger()
env_settings = get_env_settings()


def extract_fields(data: Dict[str, Any], field_configs: List[Dict[str, Union[str, None]]]) -> Tuple[str, List[Tuple[str, Any]]]:
    """
    Extract fields from WIZQA data based on dynamic field configurations.
    Args:
        data (Dict[str, Any]): The WIZQA JSON data dictionary
        field_configs (List[Dict[str, Union[str, None]]]): List of field configuration dictionaries.
                      Each config should have:
                      - 'name' (str): Field name for identification
                      - 'path' (str): Dot-separated path to the data (e.g., 'data.search.gossip')
                      - 'item_key' (str | None): Key to extract from dict items, or None for direct values
    Returns:
        Tuple[str, List[Tuple[str, Any]]]: (query, results) where:
            - query (str): The search query from data.search.query
            - results (List[Tuple[str, Any]]): List of (field_name, extracted_value) tuples
    """
    results = []
    query = data.get("data", {}).get("search", {}).get("query", "")
    # Process each field configuration
    for config in field_configs:
        try:
            field_name = config['name']
            path = config['path']
            item_key = config.get('item_key')
            # Navigate to the specified path
            current = data
            for part in path.split('.'):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    current = None
                    break
            if current is None:
                continue
            # Convert single values to list for uniform handling
            items_to_process = current if isinstance(current, list) else [current]
            # Process each item in the list
            for item in items_to_process:
                if item_key is None:
                    results.append((field_name, item))
                elif isinstance(item, dict) and item_key in item:
                    # Extract specific key from dict items
                    results.append((field_name, item[item_key]))
        except (KeyError, TypeError, AttributeError) as e:
            # Log the error but continue processing other fields
            logger.warning(f"Error extracting field '{config.get('name', 'unknown')}': {e}")
            continue
    return query, results
