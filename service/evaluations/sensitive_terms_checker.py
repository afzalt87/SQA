import os
import json
import time
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from service.llm import Llm, load_prompt, fill_prompt
from service.utils.config import get_logger, get_env_settings
from service.utils.csv_validator import SensitiveTermsCSVValidator

logger = get_logger()

# Initialize LLM client once
llm = Llm()

# Load prompt templates once
system_prompt = load_prompt("resource/prompt/system/sensitive_terms_system.txt")
user_template = load_prompt("resource/prompt/user/sensitive_terms_user.txt")

# Load env settings for checker configuration
_ENV = get_env_settings() or {}
_CHECKER_CFG = (_ENV.get("sensitive_terms_checker") or {})
_BATCH_SIZE = int(_CHECKER_CFG.get("batch_size", 10))
_MAX_WORKERS = int(_CHECKER_CFG.get("max_workers", 5))
_CONF_THRESHOLD = float(_CHECKER_CFG.get("confidence_threshold", 0.7))
_ALLOWED_CATEGORIES = set(_CHECKER_CFG.get("categories", [
    "natural_disasters",
    "violence_events",
    "terrorism",
    "civil_unrest",
    "health_crises",
    "sensitive_deaths"
]))

# Sensitive categories with examples and severity mapping
SENSITIVE_CATEGORIES = {
    "natural_disasters": {
        "description": "Natural disasters including earthquakes, hurricanes, floods, wildfires",
        "examples": ["earthquake victims", "hurricane damage", "flood casualties", "wildfire evacuation"],
        "severity": "high"
    },
    "violence_events": {
        "description": "School shootings, mass violence events, and related incidents",
        "examples": ["school shooting", "mass shooting", "active shooter", "gun violence"],
        "severity": "critical"
    },
    "terrorism": {
        "description": "Terrorist attacks and security incidents",
        "examples": ["terrorist attack", "bombing", "extremist violence", "security breach"],
        "severity": "critical"
    },
    "civil_unrest": {
        "description": "Civil unrest, riots, protests, and social conflicts",
        "examples": ["riots", "violent protests", "civil war", "political violence"],
        "severity": "high"
    },
    "health_crises": {
        "description": "Medical emergencies, pandemics, and health crises",
        "examples": ["pandemic deaths", "medical emergency", "disease outbreak", "health crisis"],
        "severity": "high"
    },
    "sensitive_deaths": {
        "description": "Celebrity deaths, tragic accidents, or other sensitive mortality events",
        "examples": ["celebrity death", "tragic accident", "fatal incident", "mass casualties"],
        "severity": "medium"
    }
}


class SensitiveTermsChecker:
    """Checks queries for sensitive terms inappropriate for advertising."""

    def __init__(self, batch_size: Optional[int] = None, max_workers: Optional[int] = None):
        self.batch_size = int(batch_size or _BATCH_SIZE)
        self.max_workers = int(max_workers or _MAX_WORKERS)
        self.results = []

    def _parse_json_safely(self, text: str) -> Optional[Dict]:
        """Try to parse JSON; if text has extra content, extract the first JSON object via regex."""
        try:
            return json.loads(text)
        except Exception:
            # Attempt to extract JSON object
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return None
            return None

    def check_single_query(self, query: str, context: str = "") -> Dict:
        """
        Check a single query for sensitive content using GPT-4.1 (configurable).
        Returns a normalized dict with is_sensitive/category/confidence/reasoning/error fields.
        """
        try:
            prompt_vars = {
                "query": query,
                "context": context if context else "No context available",
                "categories": json.dumps(SENSITIVE_CATEGORIES, indent=2)
            }
            user_prompt = fill_prompt(user_template, prompt_vars)

            # Retry with exponential backoff
            attempts = 3
            backoff = 1.0
            response_text = None
            for attempt in range(attempts):
                response_text = llm.call_with_text(
                    system_prompt,
                    user_prompt,
                    model="gpt-5"
                )
                if response_text is not None:
                    break
                time.sleep(backoff)
                backoff *= 2

            if response_text is None:
                logger.error(f"LLM call failed for query: {query}")
                return {
                    "query": query,
                    "is_sensitive": False,
                    "category": None,
                    "confidence": 0.0,
                    "reasoning": "LLM call failed",
                    "error": True
                }

            parsed = self._parse_json_safely(response_text)
            if not isinstance(parsed, dict):
                logger.error(f"Failed to parse LLM response for query: {query}")
                return {
                    "query": query,
                    "is_sensitive": False,
                    "category": None,
                    "confidence": 0.0,
                    "reasoning": "Failed to parse LLM response",
                    "error": True
                }

            # Normalize and validate
            is_sensitive = bool(parsed.get("is_sensitive", False))
            category = parsed.get("category", None)
            if category not in _ALLOWED_CATEGORIES:
                category = None
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            reasoning = str(parsed.get("reasoning", ""))

            result = {
                "query": query,
                "is_sensitive": is_sensitive and confidence >= _CONF_THRESHOLD,
                "category": category,
                "confidence": confidence,
                "reasoning": reasoning,
                "error": False
            }
            logger.info(f"Checked query '{query}': sensitive={result['is_sensitive']} cat={category} conf={confidence:.2f}")
            return result

        except Exception as e:
            logger.exception(f"Error checking query '{query}': {e}")
            return {
                "query": query,
                "is_sensitive": False,
                "category": None,
                "confidence": 0.0,
                "reasoning": str(e),
                "error": True
            }
            # unreachable log (kept from a previous edit) — removing to avoid dead code

    def check_queries_batch(self, queries_with_context: List[Tuple[str, str]]) -> List[Dict]:
        results: List[Dict] = []
        logger.info(f"Submitting {len(queries_with_context)} queries to thread pool (max_workers={self.max_workers})")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_query = {
                executor.submit(self.check_single_query, query, context): (query, context)
                for query, context in queries_with_context
            }
            processed = 0
            for future in as_completed(future_to_query):
                query, context = future_to_query[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {e}")
                    result = {
                        "query": query,
                        "is_sensitive": False,
                        "category": None,
                        "confidence": 0.0,
                        "reasoning": str(e),
                        "error": True
                    }
                results.append(result)
                processed += 1
                if processed % 25 == 0:
                    logger.info(f"Progress: {processed}/{len(future_to_query)} queries evaluated")
        return results
    
    def check_all_queries(self, queries_data: Dict[str, Dict]) -> List[Dict]:
        queries_with_context = [
            (query, (data or {}).get("summary", ""))
            for query, data in queries_data.items()
        ]

        logger.info(f"Checking {len(queries_with_context)} queries for sensitive terms")

        all_results: List[Dict] = []
        for i in range(0, len(queries_with_context), self.batch_size):
            batch = queries_with_context[i:i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1} ({len(batch)} queries)")
            all_results.extend(self.check_queries_batch(batch))

        sensitive_queries = [
            r for r in all_results if r.get("is_sensitive", False)
        ]
        logger.info(f"Found {len(sensitive_queries)} sensitive queries out of {len(all_results)}")
        return sensitive_queries


def run_sensitive_terms_check(trend_data: Dict[str, Dict], override_mode: bool = False) -> List[Dict]:
    """
    Main entry point for sensitive terms checking.
    
    Args:
        trend_data: Dictionary of trending queries with their data
        override_mode: If True, read queries from CSV instead of trend_data
        
    Returns:
        List of sensitive query issues
    """
    checker = SensitiveTermsChecker()
    
    if override_mode:
        # Load queries from override CSV
        override_path = "resource/override_sensitive_terms.csv"
        logger.info(f"Override mode: Loading queries from {override_path}")
        
        try:
            df = pd.read_csv(override_path)
            if 'query' not in df.columns:
                logger.error(f"CSV file must have 'query' column. Found columns: {df.columns.tolist()}")
                return []
            
            # Convert to format expected by checker
            queries_data = {row['query']: {"summary": ""} for _, row in df.iterrows()}
            logger.info(f"Loaded {len(queries_data)} queries from override CSV")
            
        except FileNotFoundError:
            logger.error(f"Override CSV not found: {override_path}")
            return []
        except Exception as e:
            logger.error(f"Error reading override CSV: {e}")
            return []
    else:
        queries_data = trend_data
    
    # Check all queries
    sensitive_queries = checker.check_all_queries(queries_data)
    
    # Format results for reporting
    formatted_results: List[Dict] = []
    for result in sensitive_queries:
        category = result.get("category") or "unknown"
        severity = SENSITIVE_CATEGORIES.get(category, {}).get("severity", "unknown")
        formatted_results.append({
            "query": result["query"],
            "error_type": "sensitive_terms",
            "category": category,
            "severity": severity,
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
            "is_dead": "no",
            "module": "sensitive_terms_checker"
        })
    
    return formatted_results


# For backwards compatibility and testing
def check_queries_before_context(queries: List[str]) -> List[Dict]:
    """
    Check queries before context is added (legacy interface).
    
    Args:
        queries: List of query strings
        
    Returns:
        List of sensitive query issues
    """
    queries_data = {query: {"summary": ""} for query in queries}
    return run_sensitive_terms_check(queries_data)

# --- CLI entrypoint: run sensitive checker standalone (fetched terms or CSV) ---
if __name__ == "__main__":
    import os
    import sys
    import argparse
    import datetime as _dt
    import pandas as _pd

    # Allow running via: python service/evaluations/sensitive_terms_checker.py (from repo root)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from service.fetchers.trend_fetcher import generate_trends
    from service.utils.csv_validator import SensitiveTermsCSVValidator
    from service.utils.config import setup_file_logging, get_logger, get_env_settings
    setup_file_logging("logs")
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description="Sensitive terms checker (standalone)")
    parser.add_argument("--sensitive_override", action="store_true",
                        help="Run from CSV only (skip fetching)")
    parser.add_argument("--input", "-i", default="resource/override_sensitive_terms.csv",
                        help="CSV path with header 'query' (used when --sensitive_override is set or for custom CSV)")
    parser.add_argument("--out", "-o", default=None,
                        help="Output CSV path (defaults to reports/<timestamp>_sensitive_terms.csv)")
    parser.add_argument("--min_conf", type=float, default=None,
                        help="Override minimum confidence threshold just for this run")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size just for this run")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Override max workers just for this run")
    parser.add_argument("--limit", type=int, default=None,
                    help="Limit number of fetched queries to process (standalone fetched mode)")
    args = parser.parse_args()

    # Load config
    env_cfg = (get_env_settings() or {}).get("sensitive_terms_checker", {}) or {}

    # Apply optional runtime overrides
    if args.min_conf is not None:
        globals()['_CONF_THRESHOLD'] = float(args.min_conf)

    bs = int(args.batch_size or env_cfg.get("batch_size", 10))
    mw = int(args.max_workers or env_cfg.get("max_workers", 5))

    checker = SensitiveTermsChecker(batch_size=bs, max_workers=mw)

    # Determine data source: CSV override vs fetched terms
    if args.sensitive_override:
        is_valid, df, err = SensitiveTermsCSVValidator.validate_and_load(args.input)
        if not is_valid:
            logger.error(f"CSV invalid: {err}")
            sys.exit(1)
        queries_data = {row["query"]: {"summary": ""} for _, row in df.iterrows()}
        sensitive = checker.check_all_queries(queries_data)
    else:
        logger.info("Fetching raw trends (NUWA/Google/Yahoo)...")
        trends = generate_trends()
        if args.limit:
            limited = {}
            for i, (q, d) in enumerate(trends.items()):
                if i >= args.limit:
                    break
                limited[q] = d
            trends = limited
            logger.info(f"Limiting to {len(trends)} fetched queries due to --limit")
        else:
            logger.info(f"Fetched {len(trends)} queries (pre-context)")
    sensitive = checker.check_all_queries(trends)

    # Format results for report (align with main report)
    rows = []
    for result in sensitive:
        category = result.get("category") or "unknown"
        severity = SENSITIVE_CATEGORIES.get(category, {}).get("severity", "unknown")
        rows.append({
            "query": result["query"],
            "error_type": "sensitive_terms",
            "category": category,
            "severity": severity,
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
            "is_dead": "no",
            "module": "sensitive_terms_checker",
        })

    if not rows:
        rows = [{
            "query": "N/A",
            "error_type": "sensitive_terms",
            "category": "nothing to report",
            "severity": "N/A",
            "confidence": "N/A",
            "reasoning": "No sensitive terms identified in current batch",
            "is_dead": "no",
            "module": "sensitive_terms_checker",
        }]

    # Save outputif args.sensitive_override:
    out_path = args.out
    if not out_path:
        os.makedirs("reports", exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("reports", f"{stamp}_sensitive_terms.csv")
    _pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Saved sensitive checker report: {out_path}")