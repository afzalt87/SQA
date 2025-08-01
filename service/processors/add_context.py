from service.llm import Llm, load_prompt, fill_prompt
from service.utils.config import get_logger

logger = get_logger()
llm = Llm()

# Load prompts once
system_prompt = load_prompt("resource/prompt/system/add_context_system.txt")
user_template = load_prompt("resource/prompt/user/add_context_user.txt")


def build_user_prompt(term, article_titles):
    if not article_titles:
        return None
    title_block = "\n".join(f"- {title}" for title in article_titles)
    return fill_prompt(user_template, {"term": term, "headlines": title_block})


def add_context_to_trends(trends: dict) -> dict:
    summarized_trends = {}

    for term, data in trends.items():
        articles = data.get("articles", [])
        if not articles:
            data["summary"] = "NA"
            summarized_trends[term] = data
            continue

        user_prompt = build_user_prompt(term, articles[:10])
        if not user_prompt:
            data["summary"] = "NA"
            summarized_trends[term] = data
            continue

        try:
            summary = llm.call_with_text(system_prompt, user_prompt)
            data["summary"] = summary if summary and len(summary) > 5 else "NA"
            logger.info(f"[✓] {term}: {summary}")
        except Exception as e:
            logger.warning(f"[X] Failed to summarize '{term}': {e}")
            data["summary"] = "NA"

        summarized_trends[term] = data

    return summarized_trends
