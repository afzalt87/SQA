from service.llm import Llm, load_prompt, fill_prompt
from service.utils.config import get_logger
from service.utils.data_utils import iter_wizqa_json_files
from service.processors.filter_resource import extract_fields

logger = get_logger()
llm = Llm()

# Load prompt templates once
system_prompt = load_prompt("resource/prompt/system/kg_relevance_system.txt")
user_template = load_prompt("resource/prompt/user/kg_relevance_user.txt")


def check_kg_match(query: str, title: str, subtitle: str = "", description: str = "") -> bool:
    logger.info(f"Checking KG match for query: {query} | title: {title} | subtitle: {subtitle} | description: {description}")
    user_prompt = fill_prompt(user_template, {
        "query": query,
        "title": title,
        "subtitle": subtitle,
        "description": description
    })

    answer = llm.call_with_text(system_prompt, user_prompt)
    if answer is None:
        return False

    if answer.lower() == "yes":  # "yes" means mismatch
        logger.info(f"KG mismatch detected for query: {query} | title: {title} | subtitle: {subtitle} | description: {description}")
        return True
    return False


def run_kg_mismatch_check(wizqa_path: str) -> list:
    results = []
    kg_checked = 0
    kg_missing = 0
    kg_skipped = 0

    for filename, filepath, data in iter_wizqa_json_files(wizqa_path):
        try:
            # Extract all needed fields at once using extract_fields
            field_configs = [
                {'name': 'query', 'path': 'data.search.query', 'item_key': None},
                {'name': 'kgPeople', 'path': 'data.search.data.kgPeople', 'item_key': None},
                {'name': 'title', 'path': 'data.search.data.kgPeople.kgHeader.data.title.txt', 'item_key': None},
                {'name': 'subtitle', 'path': 'data.search.data.kgPeople.kgHeader.data.title.subTxt', 'item_key': None},
                {'name': 'description', 'path': 'data.search.data.kgPeople.kgDescription.data.description', 'item_key': None}
            ]
            query, field_results = extract_fields(data, field_configs)
            # Convert results to a dict for easy access
            extracted = {name: value for name, value in field_results}
            if 'kgPeople' not in extracted or not extracted['kgPeople']:
                kg_missing += 1
                continue

            title = extracted.get('title', "")
            subtitle = extracted.get('subtitle', "")
            description = extracted.get('description', "")

            if not (title or subtitle or description):
                kg_skipped += 1
                logger.warning(f"⚠️ KGPeople present but missing display fields for query: {query}")
                continue

            kg_checked += 1

            if check_kg_match(query, title, subtitle, description):
                offending_string = " | ".join(filter(None, [title, subtitle, description]))
                results.append({
                    "query": query,
                    "module": "kgPeople",
                    "offending_string": offending_string,
                    "matched_token": "llm_irrelevant",
                    "category": "mismatch",
                    "error_type": "kg mismatch",
                    "is_dead": "no"
                })

        except Exception as e:
            logger.warning(f"⚠️ Error processing KG People in {filename}: {e}")

    logger.info(f"✅ KG check complete. Evaluated: {kg_checked} | Skipped: {kg_skipped} | Missing: {kg_missing}")
    return results
