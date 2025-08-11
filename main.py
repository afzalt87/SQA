import os
import datetime
import time
import json
import pandas as pd
import asyncio
import aiofiles
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import multiprocessing

from service.fetchers.srp_fetcher import fetch_wizqa
from service.utils.config import get_logger, setup_file_logging, get_log_file_path
from service.utils.data_utils import iter_wizqa_json_files
from service.fetch_pipeline import run_pipeline
from service.fetchers.trend_fetcher import generate_trends
from service.processors.filter_resource import extract_fields
from service.evaluations.sa_blocklist import scan_json_files
from service.evaluations.death_check import run_death_check
from service.evaluations.kgbsport_check import run_kgbsport_check
from service.evaluations.sa_relevance import check_relevance_pair
from service.evaluations.kg_relevance import run_kg_mismatch_check
from service.evaluations.trending_searches import run_trending_searches
from service.evaluations.sensitive_terms_checker import run_sensitive_terms_check

log_file = setup_file_logging("logs")
logger = get_logger()

# Config
WIZQA_DIR = "data/wizqa"
BLOCKLIST_PATH = "resource/sa_blocklist.json"

# Concurrent processing configuration
MAX_CONCURRENT_WIZQA = 10  # We can limit concurrent WIZQA fetches to 10 to avoid API overload
MAX_CONCURRENT_FILES = 20  # We can limit concurrent file processing to 20 to avoid memory issues
USE_CONCURRENT = True  # Toggle to easily switch between concurrent and sequential

# Trending searches configuration
ENABLE_TRENDING_SEARCHES = False  # Toggle to enable/disable trending searches analysis
TRENDING_SEARCH_QUERY = "news"  # Default query for trending searches

ENABLE_SENSITIVE_TERMS = True  # Toggle to enable/disable sensitive terms check

class ConcurrentProcessor:
    """Handles concurrent processing with proper resource management."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(
            max_workers=min(32, (multiprocessing.cpu_count() or 1) * 4)
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)


def run_trend_pipeline():
    """Run the full SQA trend pipeline and return all issues found."""
    logger.info("📈 [1/8] Running trend pipeline (with context)")
    trend_data = run_pipeline()
    queries = list(trend_data.keys())
    logger.info(f"Retrieved {len(queries)} queries")
    return trend_data, queries

## Remove old stubbed sensitive terms function (handled by evaluator)

async def fetch_wizqa_batch_async(queries_batch: List[str], semaphore: asyncio.Semaphore, timestamp: str, woeid_list=None):
    """Fetch WIZQA for a batch of queries with rate limiting."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            if woeid_list:
                await loop.run_in_executor(executor, fetch_wizqa, queries_batch, timestamp, woeid_list)
            else:
                await loop.run_in_executor(executor, fetch_wizqa, queries_batch, timestamp)


async def fetch_and_save_wizqa_concurrent(queries: List[str], woeid_list=None) -> str:
    """Fetch WIZQA responses for all queries concurrently."""
    logger.info(f"🧠 [2/8] Fetching WIZQA responses (concurrent mode for {len(queries)} queries)")
    
    # Generate timestamp once for all batches
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Splitting queries into batches to avoid overwhelming the API
    batch_size = 5
    query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    
    # Split woeid_list into corresponding batches if it exists
    woeid_batches = None
    if woeid_list:
        woeid_batches = [woeid_list[i:i + batch_size] for i in range(0, len(woeid_list), batch_size)]
    
    # Creating semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WIZQA)

    # Creating tasks for all batches
    tasks = []
    for i, batch in enumerate(query_batches):
        woeid_batch = woeid_batches[i] if woeid_batches else None
        tasks.append(fetch_wizqa_batch_async(batch, semaphore, timestamp, woeid_batch))
    
    await asyncio.gather(*tasks)
    
    wizqa_path = os.path.join(WIZQA_DIR, timestamp)
    return wizqa_path


def fetch_and_save_wizqa(queries, woeid_list=None):
    """Fetch WIZQA responses for all queries (with optional concurrent mode)."""
    if USE_CONCURRENT and len(queries) > 5:
        return asyncio.run(fetch_and_save_wizqa_concurrent(queries, woeid_list))
    else:
        logger.info("🧠 [2/8] Fetching WIZQA responses")
        # Generate timestamp once
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if woeid_list:
            fetch_wizqa(queries, timestamp, woeid_list)
        else:
            fetch_wizqa(queries, timestamp)
        wizqa_path = os.path.join(WIZQA_DIR, timestamp)
        return wizqa_path


def scan_blocklist(wizqa_path):
    """Scan WIZQA results for blocklist matches."""
    logger.info("🧹 [3/8] Running SA blocklist scan")
    sa_results = scan_json_files(wizqa_path, BLOCKLIST_PATH)
    logger.info(f"🔍 Blocklist matches found: {len(sa_results)}")
    for r in sa_results:
        logger.info(f"⚠️  Query: {r['query']}, Category: {r['category']}, "
                    f"Match: '{r['matched_token']}' in module {r['module']}")
    return sa_results


def scan_death_context(trend_data, wizqa_path):
    """Scan for death-related context in trend data."""
    logger.info("💀 [4/8] Running death context check")
    death_results = run_death_check(trend_data, wizqa_path)
    logger.info(f"☠️  Death-related context matches found: {len(death_results)}")
    return death_results


def scan_kgbsport_context(trend_data, wizqa_path):
    """Scan those WIZQA results that triggered kgBrowseSports"""
    logger.info("🏈 [5/8] Running kgbsport context check")
    kgbsport_results = run_kgbsport_check(trend_data, wizqa_path)
    return kgbsport_results


async def process_relevance_file_async(filepath: str, executor: ThreadPoolExecutor) -> List[Dict]:
    """Process a single file for relevance check asynchronously."""
    results = []
    filename = os.path.basename(filepath)
    
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
        
        # Use extract_fields for consistent data extraction
        field_configs = [
            {'name': 'peopleAlsoAsk', 'path': 'data.search.data.peopleAlsoAsk.peopleAlsoAsk.data.list', 'item_key': 'title'}
        ]
        query, results_data = extract_fields(data, field_configs)
        
        suggestions = [value for name, value in results_data if name == 'peopleAlsoAsk']
        
        loop = asyncio.get_event_loop()
        for suggestion in suggestions:
            is_irrelevant = await loop.run_in_executor(
                executor,
                check_relevance_pair,
                query,
                suggestion
            )
            if is_irrelevant:
                results.append({
                    "query": query,
                    "module": "peopleAlsoAsk",
                    "offending_string": suggestion,
                    "matched_token": "llm_irrelevant",
                    "category": "off_topic",
                    "error_type": "relevance",
                    "is_dead": "no"
                })
                
        if results:
            logger.info(f"[🔍] Found {len(results)} relevance issues in {filename}")
            
    except Exception as e:
        logger.warning(f"⚠️ Error processing {filename}: {e}")
    
    return results


async def check_relevance_concurrent(wizqa_path: str) -> List[Dict]:
    """Run relevance check on WIZQA suggestions concurrently."""
    logger.info("🧠 [6/8] Running relevance check on suggestions (concurrent)")
    
    json_files = [
        os.path.join(wizqa_path, f) 
        for f in os.listdir(wizqa_path) 
        if f.endswith(".json")
    ]
    
    relevance_results = []
    
    # Processing files concurrently with semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)
    
    with ConcurrentProcessor() as processor:
        async def process_with_semaphore(filepath):
            async with semaphore:
                return await process_relevance_file_async(filepath, processor.executor)
        
        tasks = [process_with_semaphore(f) for f in json_files]
        results_lists = await asyncio.gather(*tasks)
        
        relevance_results = [item for sublist in results_lists for item in sublist]
    
    logger.info(f"🔍 Relevance mismatches found: {len(relevance_results)}")
    return relevance_results


def check_relevance(wizqa_path):
    """Run relevance check on WIZQA suggestions."""
    if USE_CONCURRENT:
        return asyncio.run(check_relevance_concurrent(wizqa_path))
    else:
        logger.info("🧠 [6/8] Running relevance check on suggestions")
        relevance_results = []
        for filename, filepath, data in iter_wizqa_json_files(wizqa_path):
            logger.info(f"[🔍] Processing file: {filename}")
            try:
                # Use extract_fields for consistent data extraction
                field_configs = [
                    {'name': 'peopleAlsoAsk', 'path': 'data.search.data.peopleAlsoAsk.peopleAlsoAsk.data.list', 'item_key': 'title'}
                ]
                query, results = extract_fields(data, field_configs)
                
                suggestions = [value for name, value in results if name == 'peopleAlsoAsk']
                for suggestion in suggestions:
                    if check_relevance_pair(query, suggestion):
                        relevance_results.append({
                            "query": query,
                            "module": "peopleAlsoAsk",
                            "offending_string": suggestion,
                            "matched_token": "llm_irrelevant",
                            "category": "off_topic",
                            "error_type": "relevance",
                            "is_dead": "no"
                        })
            except Exception as e:
                logger.warning(f"⚠️ Error processing {filename}: {e}")
        logger.info(f"🔍 Relevance mismatches found: {len(relevance_results)}")
        return relevance_results


def check_kg_mismatch(wizqa_path):
    """Run KG People match check."""
    logger.info("👤 [7/8] Running KG People match check")
    kg_mismatch_results = run_kg_mismatch_check(wizqa_path)
    logger.info(f"🔍 KG mismatches found: {len(kg_mismatch_results)}")
    return kg_mismatch_results


def run_trending_searches_check():
    """Run trending searches module analysis."""
    if not ENABLE_TRENDING_SEARCHES:
        logger.info("📊 Trending searches analysis is disabled")
        return [], pd.DataFrame()
    
    logger.info("📊 [8/8] Running trending searches module analysis")
    try:
        trending_issues, trending_report = run_trending_searches(
            query=TRENDING_SEARCH_QUERY,
            use_web_search=True
        )
        logger.info(f"🔍 Trending searches issues found: {len(trending_issues)}")
        return trending_issues, trending_report
    except Exception as e:
        logger.error(f"Error running trending searches analysis: {e}")
        return [], pd.DataFrame()

## Remove obsolete single-query stub helpers for sensitive terms


def save_issues_report(all_issues, latest_wizqa_folder, trending_report_df=None):
    """Save all issues to CSV reports."""
    # Save main issues report
    if all_issues:
        logger.debug(f"Saving {len(all_issues)} issues to report")        
        try:
            df = pd.DataFrame(all_issues)
            column_order = [
                "error_type", "query", "module", "offending_string",
                "matched_token", "category", "is_dead"
            ]
            # Add optional columns if they exist
            if "position" in df.columns:
                column_order.append("position")
            if "source" in df.columns:
                column_order.append("source")
            if "justification" in df.columns:
                column_order.append("justification")
            # Sensitive terms optional columns
            if "severity" in df.columns:
                column_order.append("severity")
            if "confidence" in df.columns:
                column_order.append("confidence")
            if "reasoning" in df.columns:
                column_order.append("reasoning")
            
            df = df[[col for col in column_order if col in df.columns]]
            csv_name = f"{latest_wizqa_folder}.csv"
            output_path = os.path.join("reports", csv_name)
            os.makedirs("reports", exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"📄 Combined results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error creating DataFrame from issues: {e}")
            logger.error(f"All issues sample: {all_issues[:5] if len(all_issues) > 5 else all_issues}")
            raise
    else:
        logger.info("✅ No issues found in blocklist, death check, relevance, or KG mismatch.")
    
    # Save trending searches detailed report if enabled
    if ENABLE_TRENDING_SEARCHES and trending_report_df is not None and not trending_report_df.empty:
        trending_csv_name = f"{latest_wizqa_folder}_trending_searches.csv"
        trending_output_path = os.path.join("reports", trending_csv_name)
        trending_report_df.to_csv(trending_output_path, index=False)
        logger.info(f"📄 Trending searches detailed report saved to: {trending_output_path}")

    # No-op: sensitive detail report handling is performed elsewhere when needed


def combine_all_issues(sa_results, death_results, kgbsport_results, relevance_results, kg_mismatch_results, trending_results=None, sensitive_results=None):
    """Combine all issue lists into a single list with proper annotations."""
    all_issues = []
    
    # Debug: Check what we're getting
    logger.info(f"🔄 Combining evaluation results...")
    logger.info(f"📊 SA results: {len(sa_results) if hasattr(sa_results, '__len__') else 'N/A'} items")
    logger.info(f"📊 Death results: {len(death_results) if hasattr(death_results, '__len__') else 'N/A'} items")
    logger.info(f"📊 KGBSport results: {len(kgbsport_results) if hasattr(kgbsport_results, '__len__') else 'N/A'} items")
    logger.info(f"📊 Relevance results: {len(relevance_results) if hasattr(relevance_results, '__len__') else 'N/A'} items")
    logger.info(f"📊 KG mismatch results: {len(kg_mismatch_results) if hasattr(kg_mismatch_results, '__len__') else 'N/A'} items")
    logger.info(f"📊 Sensitive results: {len(sensitive_results) if hasattr(sensitive_results, '__len__') else 'N/A'} items")
    # Debug: Show the actual kg_mismatch_results content
    if kg_mismatch_results:
        logger.info(f"🔍 KG mismatch results content: {kg_mismatch_results}")
    
    # Process SA results
    if sa_results:
        for r in sa_results:
            if isinstance(r, dict):
                r["error_type"] = "SA blocklist"
                r["is_dead"] = "no"
                all_issues.append(r)
            else:
                logger.error(f"Invalid SA result format: {type(r)} - {r}")
    
    # Process other results - ensure they're lists of dicts
    for result_list, name in [
        (death_results, "death"),
        (kgbsport_results, "kgbsport"), 
        (relevance_results, "relevance"),
        (kg_mismatch_results, "kg_mismatch")
    ]:
        if result_list:
            for i, item in enumerate(result_list):
                if isinstance(item, dict):
                    all_issues.append(item)
                elif isinstance(item, tuple):
                    logger.error(f"Found tuple in {name} results at index {i}: {item} - SKIPPING")
                else:
                    logger.error(f"Invalid {name} result format at index {i}: {type(item)} - {item}")
    
    # Process trending results
    if trending_results:
        for item in trending_results:
            if isinstance(item, dict):
                all_issues.append(item)
            else:
                logger.error(f"Invalid trending result format: {type(item)} - {item}")
    
    # Append sensitive results if provided
    if sensitive_results:
        for item in sensitive_results:
            if isinstance(item, dict):
                all_issues.append(item)
            else:
                logger.error(f"Invalid sensitive result format: {type(item)} - {item}")
    logger.info(f"Combined {len(all_issues)} total issues from all evaluations")
    return all_issues


async def run_evaluations_concurrent(wizqa_path: str, trend_data: Dict) -> Tuple:
    """Run all evaluation steps concurrently."""
    logger.info("🚀 Running evaluations in parallel...")
    
    loop = asyncio.get_event_loop()
    
    with ConcurrentProcessor() as processor:
        sa_task = loop.run_in_executor(
            processor.executor, scan_blocklist, wizqa_path
        )
        death_task = loop.run_in_executor(
            processor.executor, scan_death_context, trend_data, wizqa_path
        )
        kgbsport_task = loop.run_in_executor(
            processor.executor, scan_kgbsport_context, trend_data, wizqa_path
        )
        kg_task = loop.run_in_executor(
            processor.executor, check_kg_mismatch, wizqa_path
        )
        
        # Add trending searches task if enabled
        if ENABLE_TRENDING_SEARCHES:
            trending_task = loop.run_in_executor(
                processor.executor, run_trending_searches_check
            )
        
        relevance_task = check_relevance_concurrent(wizqa_path)
        
        if ENABLE_TRENDING_SEARCHES:
            sa_results, death_results, kgbsport_results, kg_results, (trending_results, trending_report) = await asyncio.gather(
                sa_task, death_task, kgbsport_task, kg_task, trending_task
            )
        else:
            sa_results, death_results, kgbsport_results, kg_results = await asyncio.gather(
                sa_task, death_task, kgbsport_task, kg_task
            )
            trending_results, trending_report = [], pd.DataFrame()
        
        relevance_results = await relevance_task
    
    return sa_results, death_results, kgbsport_results, relevance_results, kg_results, trending_results, trending_report


def run_all_evaluations(wizqa_path: str, trend_data: Dict) -> Tuple:
    """Run all evaluation steps either concurrently or sequentially based on USE_CONCURRENT flag."""
    if USE_CONCURRENT:
        logger.info("🚀 Running evaluations in parallel...")
        return asyncio.run(run_evaluations_concurrent(wizqa_path, trend_data))
    else:
        logger.info("🔄 Running evaluations sequentially...")
        sa_results = scan_blocklist(wizqa_path)
        death_results = scan_death_context(trend_data, wizqa_path)
        kgbsport_results = scan_kgbsport_context(trend_data, wizqa_path)
        relevance_results = check_relevance(wizqa_path)
        kg_mismatch_results = check_kg_mismatch(wizqa_path)
        
        # Respect ENABLE_TRENDING_SEARCHES flag in sequential mode too
        if ENABLE_TRENDING_SEARCHES:
            trending_results, trending_report = run_trending_searches_check()
        else:
            trending_results, trending_report = [], pd.DataFrame()
            
        return sa_results, death_results, kgbsport_results, relevance_results, kg_mismatch_results, trending_results, trending_report


def process(override_trend=False, override_sensitive=False, check_sensitive_terms: bool = True, sensitive_override: bool = False):
    """
    Main processing function that:
    1. Generate trending queries.
    2. Fetch WIZQA responses for each query.
    3. Run all evaluations (blocklist, death, kgbsport, relevance, KG mismatch, trending).
    4. Combine and save all issues to reports.
    """
    
    sensitive_results: List[Dict] = []

    # Fast path: run ONLY sensitive terms from CSV and skip trend fetching/WIZQA
    if sensitive_override:
        logger.info("🚨 [0/8] Sensitive override mode: running sensitive checker from CSV and skipping trend fetching")
        sensitive_results = run_sensitive_terms_check({}, override_mode=True)
        if not sensitive_results:
            sensitive_results = [{
                "query": "N/A",
                "error_type": "sensitive_terms",
                "category": "nothing to report",
                "severity": "N/A",
                "confidence": "N/A",
                "reasoning": "No sensitive terms identified in current batch",
                "module": "sensitive_terms_checker",
                "is_dead": "no"
            }]
        # Use a timestamp label for the output filename
        latest_wizqa_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        all_issues = combine_all_issues([], [], [], [], [], [], sensitive_results)
        save_issues_report(all_issues, latest_wizqa_folder, None)
        return all_issues
    
    # Step 1: Generate trends or load from override CSV
    if override_trend:
        logger.info("📈 [1/8] Override mode: Reading queries from resource/override.csv")
        try:
            override_csv_path = os.path.join("resource", "override.csv")
            if not os.path.exists(override_csv_path):
                raise FileNotFoundError(f"Override CSV file not found: {override_csv_path}")
            
            df = pd.read_csv(override_csv_path)
            
            if 'queries' not in df.columns:
                raise ValueError("'queries' column not found in override.csv")
            
            # Get valid rows (non-null queries)
            valid_mask = df['queries'].notna()
            queries = df.loc[valid_mask, 'queries'].tolist()
            
            # Get corresponding WOEIDs if column exists
            woeid_list = None
            if 'woeid' in df.columns:
                woeid_list = df.loc[valid_mask, 'woeid'].tolist()
                # Convert NaN to None and float to int for consistency
                woeid_list = [int(w) if pd.notna(w) else None for w in woeid_list]
                logger.info(f"📄 Found {sum(1 for w in woeid_list if w is not None)} WOEIDs for {len(queries)} queries")
                
                # Log each query-woeid pair for transparency
                for query, woeid in zip(queries, woeid_list):
                    if woeid is not None:
                        logger.info(f"🔗 Query: '{query}' → WOEID: {woeid}")
                    else:
                        logger.info(f"🔗 Query: '{query}' → WOEID: None")
            
            trend_data = {}
            logger.info(f"📄 Loaded {len(queries)} queries from resource/override.csv")
            
        except Exception as e:
            logger.error(f"❌ Error reading override CSV: {e}")
            raise
    else:
        # Get raw trends to enable sensitive check pre-context
        raw_trends = generate_trends()
        queries = list(raw_trends.keys())
        woeid_list = None
        # Optionally run sensitive terms before context
        if check_sensitive_terms and ENABLE_SENSITIVE_TERMS and not sensitive_override:
            logger.info("🚨 [1.0/8] Running sensitive terms check on raw trends (pre-context)")
            # Build minimal trend_data for checker compatibility
            trend_map = {q: {"summary": ""} for q in queries}
            sensitive_results = run_sensitive_terms_check(trend_map, override_mode=False)
        # Proceed with context pipeline
        trend_data = run_pipeline()

    # If override-sensitive flag is set, run from CSV
    if sensitive_override:
        logger.info("🚨 [1.5/8] Running sensitive terms check (override CSV mode)")
        sensitive_results = run_sensitive_terms_check({}, override_mode=True)

    # Ensure a placeholder row when there is nothing to report
    if check_sensitive_terms and ENABLE_SENSITIVE_TERMS and not sensitive_results:
        sensitive_results = [{
            "query": "N/A",
            "error_type": "sensitive_terms",
            "category": "nothing to report",
            "severity": "N/A",
            "confidence": "N/A",
            "reasoning": "No sensitive terms identified in current batch",
            "module": "sensitive_terms_checker",
            "is_dead": "no"
        }]
    
    # Step 2: Fetch WIZQA responses
    if override_trend:
        wizqa_path = fetch_and_save_wizqa(queries, woeid_list)
    else:
        wizqa_path = fetch_and_save_wizqa(queries)
    latest_wizqa_folder = os.path.basename(wizqa_path)
 
    
    # Step 3: Run all evaluations
    evaluation_results = run_all_evaluations(wizqa_path, trend_data)
    logger.debug(f"Evaluation results type: {type(evaluation_results)}, length: {len(evaluation_results)}")
    
    # Unpack results safely
    if len(evaluation_results) >= 6:
        sa_results, death_results, kgbsport_results, relevance_results, kg_mismatch_results, trending_results, trending_report = evaluation_results
    else:
        logger.error(f"Expected 7 results, got {len(evaluation_results)}: {evaluation_results}")
        return []
    
    # Step 4: Combine and save results
    logger.info("📋 Start combining all evaluation results")
    all_issues = combine_all_issues(
        sa_results, death_results, kgbsport_results, relevance_results, kg_mismatch_results, trending_results, sensitive_results
    )
    logger.info(f"📋 Finished combining results - total issues: {len(all_issues)}")
    save_issues_report(all_issues, latest_wizqa_folder, trending_report)
    
    return all_issues


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SQA trend analysis pipeline")
    parser.add_argument("--override_trend", action="store_true",
                        help="Skip trend pipeline and read queries from resource/override.csv")
    parser.add_argument("--check_sensitive_terms", action="store_true",
                        help="Enable sensitive terms checking on raw trends before context")
    parser.add_argument("--sensitive_override", action="store_true",
                        help="Use override CSV for sensitive terms instead of trend data")
    parser.add_argument("--no_check_sensitive_terms", action="store_false",
                        dest="check_sensitive_terms", help="Disable sensitive terms checking")
    parser.set_defaults(check_sensitive_terms=True)

    args = parser.parse_args()
    
    dt = datetime.datetime.now().isoformat()
    logger.info(f"=== New run started at {dt} ===")
    logger.info(f"🚀 Concurrent processing: {'ENABLED' if USE_CONCURRENT else 'DISABLED'}")
    logger.info(f"📊 Trending searches: {'ENABLED' if ENABLE_TRENDING_SEARCHES else 'DISABLED'}")
    
    if args.override_trend:
        logger.info("🔄 Override mode enabled: Using queries from resource/override.csv")
    
    start_time = time.time()
    result = process(
        override_trend=args.override_trend,
        check_sensitive_terms=args.check_sensitive_terms,
        sensitive_override=args.sensitive_override,
    )
    elapsed = time.time() - start_time
    
    logger.info(f"=== Run finished in {elapsed:.2f} seconds ===")
    
    if USE_CONCURRENT:
        estimated_sequential_time = elapsed * 4
        logger.info(f"⚡ Estimated time saved: {estimated_sequential_time - elapsed:.2f} seconds")