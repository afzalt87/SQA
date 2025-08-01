import os
from service.utils.config import get_logger


logger = get_logger()


def load_all_blocklists(blocklist_dir="resource/blocklist"):
    """
    Load all blocklist files from the blocklist
    directory into a dictionary.
    Returns: dict {category: set(words)}
    """
    blocklists = {}
    try:
        for fname in os.listdir(blocklist_dir):
            if fname.endswith(".txt"):
                category = fname.replace(".txt", "")
                with open(os.path.join(blocklist_dir, fname), "r") as f:
                    words = set(
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    )
                    blocklists[category] = words
    except Exception as e:
        logger.exception(f"Error loading blocklists: {e}")
    return blocklists


def detect_blocklist(text, blocklists):
    """
    Detects if any blocklist word/phrase is present
    in the given text (case-insensitive).
    Returns: dict {category: [matches]}
    """
    matches = {}
    try:
        if text is not None:
            text_lower = text.lower()
            for category, words in blocklists.items():
                found = [w for w in words if w.lower() in text_lower]
                if found:
                    matches[category] = found
    except Exception as e:
        logger.exception(f"Error during blocklist detection: {e}")
    return matches
