import os
import re
import pandas as pd
from service.processors.filter_resource import extract_fields
from service.utils.config import get_logger
from service.utils.data_utils import iter_wizqa_json_files, load_json_file

logger = get_logger()


def load_blocklist(path):
    logger.info(f"\n[📄] Loading blocklist from: {path}")
    blocklist = load_json_file(path)
    if blocklist is not None:
        logger.info(f"[✓] Loaded {len(blocklist)} categories from blocklist\n")
        return blocklist
    else:
        logger.error(f"[✗] Failed to load blocklist from: {path}")
        return {}


def tokenize_match(text, tokens):
    text_lower = text.lower()
    matches = []
    for token in tokens:
        if " " in token:
            if token in text_lower:
                matches.append(token)
        else:
            pattern = r'\b' + re.escape(token) + r'\b'
            if re.search(pattern, text_lower):
                matches.append(token)
    return matches


def scan_json_files(json_dir, blocklist_path):
    blocklist = load_blocklist(blocklist_path)
    all_tokens = {cat: entry["tokens"] for cat, entry in blocklist.items()}

    results = []
    logger.info(f"[📂] Scanning JSON files in: {json_dir}\n")

    # Define the field configuration for SA blocklist scanning
    field_configs = [
        {
            'name': 'gossip',
            'path': 'data.search.gossip',
            'item_key': None  # Direct array of strings
        },
        {
            'name': 'peopleAlsoAsk',
            'path': 'data.search.data.peopleAlsoAsk.peopleAlsoAsk.data.list',
            'item_key': 'title'
        },
        {
            'name': 'alsoTrySouth',
            'path': 'data.search.data.alsoTrySouth.gossipSAT.data.list',
            'item_key': 'text'
        },
        {
            'name': 'alsoTryEast',
            'path': 'data.search.data.alsoTryEast.gossipEAT.data.list',
            'item_key': 'text'
        }
    ]

    for filename, filepath, data in iter_wizqa_json_files(json_dir):
        try:
            logger.info(f"[🔍] Processing file: {filename}")
            raw_query, extracted_results = extract_fields(data, field_configs)
            logger.info(f"     ↳ Query: {raw_query}")
            logger.info(f"     ↳ Found {len(extracted_results)} items to scan")

            for module, string in extracted_results:
                for category, tokens in all_tokens.items():
                    matches = tokenize_match(string, tokens)
                    for match in matches:
                        logger.info(
                            f"       ⚠️  MATCH: '{match}' in ({module}) → "
                            f"[{category}]"
                        )
                        results.append({
                            "file": filename,
                            "query": raw_query,
                            "module": module,
                            "offending_string": string,
                            "matched_token": match,
                            "category": category
                        })
        except Exception as e:
            logger.info(f"[ERROR reading {filename}]: {e}")

    return results


if __name__ == "__main__":
    # 🔧 Update these paths as needed (testing)
    directory = "data/wizqa/20250531_170439"
    blocklist_file = "resource/sa_blocklist.json"

    flagged = scan_json_files(directory, blocklist_file)

    if flagged:
        df = pd.DataFrame(flagged)
        folder_name = os.path.basename(os.path.normpath(directory))
        output_filename = f"{folder_name}.csv"
        output_path = os.path.join("data", output_filename)

        df.to_csv(output_path, index=False)
        logger.info(f"\n✅ Found {len(flagged)} matches. Saved to {output_path}\n")
        logger.info(df.to_string(index=False))
    else:
        logger.info("\n✅ No blocklist matches found.")
