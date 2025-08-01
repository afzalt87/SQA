# trend_fetcher.py
import time
import requests
import xml.etree.ElementTree as ET
import json
from collections import defaultdict
from service.utils.config import get_logger, get_env_settings

logger = get_logger()

# Load environment settings
env_settings = get_env_settings()

NUWA_URL = env_settings["NUWA_URL"]
GOOGLE_TRENDS_RSS = env_settings["GOOGLE_TRENDS_RSS"]
YAHOO_TRENDS_API = env_settings["YAHOO_TRENDS_API"]

# --- Params for Each Source ---
NUWA_MODULES = [
    "bingnews", "sys_news_auto", "KgPeopleRelYKC", "kgSportsTeams",
    "OnlineGames", "KgMoviesYKC", "sb-finance", "KgTv",
    "eventsKgs", "trendingNowNews"
]

"""
    "bingnews", "sys_news_auto", "KgPeopleRelYKC", "kgSportsTeams",
    "OnlineGames", "KgMoviesYKC", "sb-finance", "KgTv",
    "eventsKgs", "trendingNowNews"
"""

NUWA_PARAMS = {
    "custid": "trendy.search-signal.us",
    "hits": 100,
    "sort_by": "click",
    "times_ago": "h3",
    "timeout": 30000
}

GOOGLE_TRENDS_PARAMS = {
    "geo": "US"
}

YAHOO_TRENDS_PARAMS = {
    "category": "general",
    "locale": "en_us"
}


def default_trend_entry():
    return {
        "click_data": None,
        "thumbnail": None,
        "articles": [],
        "source": []
    }

# --- Fetchers ---


def fetch_nuwa_trends():
    trend_data = defaultdict(default_trend_entry)
    for module in NUWA_MODULES:
        logger.info(f"Fetching NUWA terms from module: {module}")
        params = NUWA_PARAMS.copy()
        params["dd_list"] = module
        try:
            response = requests.get(NUWA_URL, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            entries = data.get("results", {}).get("default", {}).get("entries", [])
            for entry in entries:
                term = entry.get("term")
                if not term:
                    continue
                key = term.strip().lower()
                total_clicks = sum(entry.get("click", []))
                new_articles = [
                    article.get("title") for article in entry.get("news_list", []) if article.get("title")
                ]
                thumbnail = next(
                    (article["thumbnail"] for article in entry.get("news_list", []) if article.get("thumbnail")),
                    None
                )
                trend_data[key]["click_data"] = trend_data[key]["click_data"] or 0
                trend_data[key]["click_data"] += total_clicks
                existing_titles = set(a.lower().strip() for a in trend_data[key]["articles"])
                for title in new_articles:
                    norm = title.lower().strip()
                    if norm not in existing_titles:
                        trend_data[key]["articles"].append(title)
                        existing_titles.add(norm)
                if not trend_data[key]["thumbnail"] and thumbnail:
                    trend_data[key]["thumbnail"] = thumbnail
                if "nuwa" not in trend_data[key]["source"]:
                    trend_data[key]["source"].append("nuwa")
        except Exception as e:
            logger.warning(f"Failed to fetch NUWA data for module '{module}': {e}")
        time.sleep(0.5)
    return trend_data


def fetch_google_trends():
    trend_data = defaultdict(default_trend_entry)
    try:
        logger.info("Fetching terms from Google Trends RSS")
        response = requests.get(GOOGLE_TRENDS_RSS, params=GOOGLE_TRENDS_PARAMS, timeout=5)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall(".//item"):
            title_elem = item.find("title")
            if not title_elem or not title_elem.text:
                continue
            term = title_elem.text.strip().lower()
            key = term
            new_articles = [
                news_item.findtext("ht:news_item_title")
                for news_item in item.findall("ht:news_item")
                if news_item.findtext("ht:news_item_title")
            ]
            thumbnail = next(
                (news_item.findtext("ht:news_item_picture")
                 for news_item in item.findall("ht:news_item")
                 if news_item.findtext("ht:news_item_picture")),
                None
            )
            existing_titles = set(a.lower().strip() for a in trend_data[key]["articles"])
            for title in new_articles:
                norm = title.lower().strip()
                if norm not in existing_titles:
                    trend_data[key]["articles"].append(title)
                    existing_titles.add(norm)
            if not trend_data[key]["thumbnail"] and thumbnail:
                trend_data[key]["thumbnail"] = thumbnail
            if "google" not in trend_data[key]["source"]:
                trend_data[key]["source"].append("google")
    except Exception as e:
        logger.warning(f"Failed to fetch Google Trends RSS: {e}")
    return trend_data


def fetch_yahoo_trends():
    trend_data = defaultdict(default_trend_entry)
    try:
        logger.info("Fetching terms from Yahoo Trending API")
        response = requests.get(YAHOO_TRENDS_API, params=YAHOO_TRENDS_PARAMS, timeout=5)
        response.raise_for_status()
        data = response.json()
        items = data.get("itemsInfo", {}).get("items", [])
        for item in items:
            raw_query = item.get("raw_query")
            if not raw_query:
                continue
            key = raw_query.strip().lower()
            thumbnail = item.get("thumbnail")
            if not trend_data[key]["thumbnail"] and thumbnail:
                trend_data[key]["thumbnail"] = thumbnail
            if "yahoo" not in trend_data[key]["source"]:
                trend_data[key]["source"].append("yahoo")
    except Exception as e:
        logger.warning(f"Failed to fetch Yahoo Trending API: {e}")
    return trend_data


# --- Aggregator ---


def generate_trends():
    try:
        final_trends = defaultdict(default_trend_entry)

        for source_fetcher in [fetch_nuwa_trends, fetch_google_trends, fetch_yahoo_trends]:
            source_data = source_fetcher()
            for term, data in source_data.items():
                final_trends[term]["click_data"] = (
                    (final_trends[term]["click_data"] or 0) + (data["click_data"] or 0)
                    if data["click_data"] is not None
                    else final_trends[term]["click_data"]
                )
                existing_titles = set(a.lower().strip() for a in final_trends[term]["articles"])
                for title in data["articles"]:
                    norm = title.lower().strip()
                    if norm not in existing_titles:
                        final_trends[term]["articles"].append(title)
                        existing_titles.add(norm)
                if not final_trends[term]["thumbnail"] and data["thumbnail"]:
                    final_trends[term]["thumbnail"] = data["thumbnail"]
                for source in data["source"]:
                    if source not in final_trends[term]["source"]:
                        final_trends[term]["source"].append(source)

        logger.info(f"Collected {len(final_trends)} unique trending terms with metadata.")
        return dict(final_trends)

    except Exception as e:
        logger.exception(f"Trend generation error: {e}")
        return {}

# --- Save JSON ---


def save_trends_to_json(trends, output_path="trends_output.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trends, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved output to {output_path}")

# --- Entry Point ---


if __name__ == "__main__":
    trends = generate_trends()
    save_trends_to_json(trends)

    for term, data in trends.items():
        logger.info(f"\nTerm: {term}")
        logger.info(f"  Clicks: {data['click_data']}")
        logger.info(f"  Thumbnail: {data['thumbnail']}")
        logger.info(f"  Articles: {data['articles']}")
        logger.info(f"  Source: {data['source']}")
