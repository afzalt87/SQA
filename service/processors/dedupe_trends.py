from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from service.utils.config import get_logger


logger = get_logger()


def build_embedding_input(term, data):
    summary = data.get("summary", "").strip().lower()
    return f"{term}: {summary}" if summary and summary != "na" else term


def deduplicate_trends(trends: dict, save_path: str = "trends_deduped.json") -> dict:
    term_keys = list(trends.keys())
    embedding_inputs = [build_embedding_input(term, trends[term]) for term in term_keys]

    logger.info(f"Generating embeddings for {len(embedding_inputs)} terms...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(embedding_inputs, normalize_embeddings=True)
    similarity_matrix = cosine_similarity(embeddings)

    # Group similar terms
    THRESHOLD = 0.7
    visited = set()
    groups = []

    for i, term in enumerate(term_keys):
        if term in visited:
            continue
        group = [term]
        visited.add(term)
        for j in range(i + 1, len(term_keys)):
            if term_keys[j] not in visited and similarity_matrix[i][j] >= THRESHOLD:
                group.append(term_keys[j])
                visited.add(term_keys[j])
        groups.append(group)

    # Merge grouped terms
    deduped_trends = {}
    for group in groups:
        primary = group[0]
        merged = {
            "click_data": 0,
            "thumbnail": trends[primary].get("thumbnail"),
            "articles": [],
            "source": [],
            "summary": trends[primary].get("summary", "NA"),
            "contextual_duplicates": group[1:]
        }

        seen_articles = set()
        seen_sources = set()

        for term in group:
            data = trends[term]

            # Clicks
            merged["click_data"] += data.get("click_data") or 0

            # Articles
            for title in data.get("articles", []):
                norm = title.strip().lower()
                if norm not in seen_articles:
                    merged["articles"].append(title)
                    seen_articles.add(norm)

            # Sources
            for src in data.get("source", []):
                if src not in seen_sources:
                    merged["source"].append(src)
                    seen_sources.add(src)

        deduped_trends[primary] = merged

    logger.info(f"Deduplicated from {len(trends)} → {len(deduped_trends)} terms")

    # --- Save to file ---
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(deduped_trends, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved deduplicated output to {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save output to {save_path}: {e}")

    return deduped_trends
