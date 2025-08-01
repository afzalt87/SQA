# kgbsport_check.py
from service.processors.filter_resource import extract_fields
from service.utils.data_utils import content_exists_in_wizqa, get_wizqa_file_path, load_json_file
from service.llm import Llm, load_prompt, fill_prompt
from service.utils.config import get_logger

logger = get_logger()

# Initialize the LLM client once
llm = Llm()

# Load prompt templates once
system_prompt = load_prompt("resource/prompt/system/kg_sports_relevance_system.txt")
user_template = load_prompt("resource/prompt/user/kg_sports_relevance_user.txt")


def missing_tabs(kgbrowse_sports_content: dict) -> bool:
    """
    Check if kgBrowseSports has missing tabs(only 1 key in the kgBrowseSports dict).
    """
    if not isinstance(kgbrowse_sports_content, dict):
        return False
    keys = list(kgbrowse_sports_content.keys())
    # Return True if no keys or only 1 key exists (indicating missing tabs)
    return len(keys) <= 1


def is_substring_match(query: str, kg_title: str) -> bool:
    """Return True if the kg_title includes the query (case-insensitive)"""
    return query.lower() in kg_title.lower()


def irrelevant(kg_title: str, query: str, summary: str = "") -> bool:
    """
    Return True if the kgBrowseSports content is irrelevant to the query.
    First checks for substring match (fast path), then falls back to LLM.
    """
    if not kg_title or not query:
        logger.debug("         → Empty title or query, marking as irrelevant")
        return True  # Consider empty titles/queries as irrelevant
    if is_substring_match(query, kg_title):
        logger.debug("         → Substring match found, marking as relevant")
        return False  # Considered relevant
    logger.debug("         → No substring match, calling LLM for relevance check")
    user_prompt = fill_prompt(user_template, {
        "query": query,
        "kg_title": kg_title,
        "summary": summary
    })
    answer = llm.call_with_text(system_prompt, user_prompt)
    if answer is None:
        logger.error(f"LLM call failed for kgBrowseSports relevance check - Query: '{query}', KG Title: '{kg_title}'")
        return False  # Default to relevant if LLM fails
    is_irrelevant = answer.lower() == "yes"
    logger.debug(f"         → LLM response: '{answer}' → {'IRRELEVANT' if is_irrelevant else 'RELEVANT'}")
    return is_irrelevant  # "yes" means irrelevant


def run_kgbsport_check(trend_data: dict, wizqa_path: str) -> list:
    """
    Flag issue when
    1. triggers kgbsport, but missing tabs
    2. triggers kgbsport, but kg content not related to the query
    """
    results = []
    logger.info(f"[🏀] Starting kgBrowseSports check for {len(trend_data)} queries in: {wizqa_path}")
    kg_found_count = 0
    missing_tabs_count = 0
    irrelevant_count = 0
    for query, info in trend_data.items():
        # First check if kgBrowseSports exists before doing any expensive extraction
        if content_exists_in_wizqa(query, wizqa_path, "data.search.data", ["kgBrowseSports"]):
            kg_found_count += 1
            logger.info(f"[🔍] Processing query with kgBrowseSports: '{query}'")
            file_path = get_wizqa_file_path(query, wizqa_path)
            data = load_json_file(file_path)
            if data is not None:
                # Only extract kgBrowseSports content when we know kgBrowseSports exists
                config = [{
                    'name': 'kgBrowseSports',
                    'path': 'data.search.data.kgBrowseSports',
                    'item_key': None  # Direct object, not extracting a specific key
                }]
                raw_query, extracted_results = extract_fields(data, config)
                # Check each extracted kgBrowseSports content for missing tabs
                for name, content in extracted_results:
                    if missing_tabs(content):
                        missing_tabs_count += 1
                        logger.warning(f"     ⚠️  MISSING TABS: '{query}'")
                        results.append({
                            "query": query,
                            "module": "kgBrowseSports",
                            "offending_string": None,
                            "matched_token": None,
                            "category": "missing modules",
                            "error_type": "missing_tabs",
                            "is_dead": "no"
                        })
                    kg_title = content.get('sbHeader', {}).get('data', {}).get('title', "")
                    summary = info.get("summary", "")
                    logger.debug(f"     ↳ KG Title: '{kg_title}'")
                    if irrelevant(kg_title, query, summary):
                        irrelevant_count += 1
                        logger.warning(f"     ⚠️  IRRELEVANT CONTENT: '{query}' → KG shows '{kg_title}'")
                        results.append({
                            "query": query,
                            "module": "kgBrowseSports",
                            "offending_string": f"query is {query}, but kgBrowseSports title is {kg_title}",
                            "matched_token": "irrelevant",
                            "category": "off topic",
                            "error_type": "relevance",
                            "is_dead": "no"
                        })
    logger.info("[✓] kgBrowseSports check completed:")
    logger.info(f"     ↳ Queries with kgBrowseSports: {kg_found_count}")
    logger.info(f"     ↳ Missing tabs issues: {missing_tabs_count}")
    logger.info(f"     ↳ Relevance issues: {irrelevant_count}")
    logger.info(f"     ↳ Total issues found: {len(results)}\n")
    return results
