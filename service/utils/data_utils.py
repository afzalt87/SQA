from typing import Dict, Tuple, Any, Optional, Iterator
import re
import os
import json
from service.utils.config import get_logger

logger = get_logger()


def sanitize_query_for_filename(query: str) -> str:
    """
    Sanitize a query string to make it safe for use as a filename.
    Args:
        query (str): The original query string
    Returns:
        str: Sanitized string safe for filenames (replaces unsafe chars with underscores)
    """
    return re.sub(r'[\\/:*?"<>|.\s]+', '_', query)


def get_wizqa_file_path(query: str, wizqa_path: str) -> str:
    """
    Get the file path for a WIZQA JSON file based on the query.
    Args:
        query (str): The search query
        wizqa_path (str): Path to the WIZQA directory
    Returns:
        str: Full path to the JSON file
    """
    safe_query = sanitize_query_for_filename(query)
    return os.path.join(wizqa_path, safe_query + ".json")


def content_exists_in_wizqa(query: str, wizqa_path: str, level: str, content: list) -> bool:
    """
    Checks if a given key exists at a specified level in the data file for the query.
    level: dot-separated string representing nested keys, e.g., 'data.search'
    content: the content to check for existence under the specified level, if passed in multiple items, only passes check when all items exist in the string representation of the current dict.
    Example:
        content_exists_in_wizqa("iain armitage", "data/wizqa/20250624_121226", 'data.search.data', ['kgPeople'])
    """
    file_path = get_wizqa_file_path(query, wizqa_path)
    found = False
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            wizqa_data = json.load(f)
        current = wizqa_data
        for part in level.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        # Check if all content exists in the string representation of the current dict
        if isinstance(current, dict) and all(item in str(current) for item in content):
            found = True
    return found


def iter_wizqa_json_files(directory: str) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """
    Iterator that yields (filename, filepath, data) for all JSON files in a directory.
    Args:
        directory (str): Path to the directory containing JSON files
    Yields:
        Tuple[str, str, Dict[str, Any]]: (filename, filepath, parsed_json_data)
    """
    logger.info(f"Scanning WIZQA JSON files in: {directory}")
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded file: {filename}")
            yield filename, filepath, data
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            logger.error(f"Error processing {filepath}: {e}")
            continue


def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely load a JSON file with error handling.
    Args:
        file_path (str): Path to the JSON file
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON data or None if file doesn't exist or has errors
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logger.info(f"Loading JSON file: {file_path}")
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None
