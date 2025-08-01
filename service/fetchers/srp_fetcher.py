import time
import os
import requests
import json
from service.utils.config import get_logger, get_env_settings
from service.utils.data_utils import sanitize_query_for_filename

logger = get_logger()
env_settings = get_env_settings()


def _fetch_screenshot_resource(query, format, output_dir, ext):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/135.0.0.0 Safari/537.36"
            )}
        query_encode = requests.utils.quote(query)
        screenshot_n_cache_api = env_settings.get("SCREENSHOT_N_CACHE_API")
        yahoo_search = env_settings.get("YAHOO_US_SRP")
        response = requests.post(
            f"{screenshot_n_cache_api}/capture_page",
            params={
                "target_url": f'{yahoo_search}?p={query_encode}',
                "format": format,
                "device": "desktop"
            },
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        task_data = response.json()
        task_id = task_data.get("task_id")
        if not task_id:
            logger.error(
                f"No task_id returned for query '{query}' (format: {format})")
            return None

        status = None
        max_retries = 30
        retries = 0
        while status not in {"completed", "failed"} and retries < max_retries:
            status_response = requests.get(
                f"{screenshot_n_cache_api}/status/{task_id}", timeout=5)
            status_response.raise_for_status()
            status_data = status_response.json()
            status = status_data.get("status")
            logger.info(
                f"{format.upper()} task {task_id} status: {status} (query: '{query}')")
            if status not in {"completed", "failed"}:
                time.sleep(1)
                retries += 1
        if status == "completed":
            output_filename = f"{output_dir}/{task_id}.{ext}"
            result_response = requests.get(
                f"{screenshot_n_cache_api}/result/{task_id}", timeout=10)
            result_response.raise_for_status()
            with open(output_filename, "wb") as f:
                f.write(result_response.content)
            logger.info(
                f"{format.upper()} saved to {output_filename} for query '{query}'")
            return output_filename
        else:
            logger.error(
                f"{format.upper()} task {task_id} failed or timed out for query '{query}'")
            return None
    except Exception as e:
        logger.exception(f"Error fetching SRP {format} for '{query}': {e}")
        return None


def fetch_png(query):
    return _fetch_screenshot_resource(query, format="png", output_dir="data/img", ext="png")


def fetch_html(query):
    return _fetch_screenshot_resource(query, format="html", output_dir="data/html", ext="html")


def fetch_and_save(query, output_folder, woeid=None):
    """
    Helper function to fetch a single WIZQA result and save it to the output folder.
    """
    try:
        logger.info(f"Fetching WIZQA for: {query}" + (f" (WOEID: {woeid})" if woeid else ""))
        wizqa_api = env_settings.get("WIZQA_API")
        
        # Build GraphQL query with or without WOEID
        params = [f'query:"{query}"']
        if woeid:
            params.insert(0, f'ipWoeId:{woeid}')
        
        graphql_query = '{search(params:{%s}){query,intl,device,timestamp,gossip,data,snapshot}}' % ','.join(params)
            
        encoded_query = requests.utils.quote(graphql_query)
        url = f"{wizqa_api}?query={encoded_query}"
        logger.info(f"Wizqa URL: {url}")

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        safe_query = sanitize_query_for_filename(query)
        file_path = os.path.join(output_folder, f"{safe_query}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"[✓] Saved: {file_path}")
        return file_path
    except Exception:
        logger.exception(f"[✗] Failed: {query}")
        return None


def fetch_wizqa(queries, timestamp, woeid_list=None):
    """
    Fetch WIZQA responses for a list of queries and save them to a single timestamped folder.
    """
    if isinstance(queries, str):
        queries = [queries]

    output_folder = os.path.join("data/wizqa", timestamp)
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Saving all WIZQA files to: {output_folder}")

    for i, query in enumerate(queries):
        woeid = woeid_list[i] if woeid_list and i < len(woeid_list) else None
        fetch_and_save(query, output_folder, woeid)
