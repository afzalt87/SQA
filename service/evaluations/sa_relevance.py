from service.llm import Llm, load_prompt, fill_prompt

# Initialize the LLM client once
llm = Llm()

# Load prompt templates once
system_prompt = load_prompt("resource/prompt/system/sa_relevance_system.txt")
user_template = load_prompt("resource/prompt/user/sa_relevance_user.txt")


def is_substring_match(query: str, suggestion: str) -> bool:
    """Return True if the suggestion includes the query (case-insensitive)"""
    return query.lower() in suggestion.lower()


def check_relevance_pair(query: str, suggestion: str) -> bool:
    """
    Return True if the suggestion is irrelevant to the query.
    First checks for substring match (fast path), then falls back to LLM.
    """
    if is_substring_match(query, suggestion):
        return False  # Considered relevant

    user_prompt = fill_prompt(user_template, {"query": query, "suggestion": suggestion})
    answer = llm.call_with_text(system_prompt, user_prompt)
    if answer is None:
        return False  # Default to relevant if LLM fails

    return answer.lower() == "yes"  # "yes" means irrelevant
