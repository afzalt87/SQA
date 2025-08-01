# death_check.py
import re
from service.utils.data_utils import content_exists_in_wizqa


def is_death_related(context: str) -> bool:
    """
    Rule-based check for signs of recent death in LLM-generated summary.
    """
    if not context or context.strip().lower() == "na":
        return False

    death_patterns = [
        r"\b(died|has died|dies at|dead at|passes away|passed away)\b",
        r"\b(obituary|funeral|memorial service)\b",
        r"\b(loses (his|her|their) battle\b)"
    ]

    text = context.lower()
    return any(re.search(pattern, text) for pattern in death_patterns)


def run_death_check(trend_data: dict, wizqa_path: str) -> list:
    """
    Check LLM-generated context for signs of recent death.
    Returns a list of flagged results.
    """
    results = []

    for query, info in trend_data.items():
        summary = info.get("summary", "")
        # Flag issue when content is dead related + triggeres kgPeople + missing Died info inside kgPeople
        if is_death_related(summary) and content_exists_in_wizqa(query, wizqa_path, "data.search.data", ["kgPeople"]) and not (content_exists_in_wizqa(query, wizqa_path, "data.search.data.kgPeople", ["Died"])):
            results.append({
                "query": query,
                "module": "context",
                "offending_string": summary,
                "matched_token": "death_pattern",
                "category": "recent_death",
                "error_type": "death check",
                "is_dead": "yes"
            })

    return results
