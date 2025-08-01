# fetch_pipeline.py
import logging
from service.fetchers.trend_fetcher import generate_trends
from service.processors.add_context import add_context_to_trends
from service.utils.config import get_logger
# from processors.dedupe_trends import deduplicate_trends

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger()


def run_pipeline():
    try:
        # Step 1: Fetch trends
        logger.info("Step 1: Fetching trends...")
        trends = generate_trends()
        logger.info(f"Fetched {len(trends)} trends.")

        # Step 2: Add context
        logger.info("Step 2: Adding context to trends...")
        trends_with_context = add_context_to_trends(trends)
        logger.info("Context added to trends.")

        # ⛔️ Skip Step 3: Deduplicate trends
        # logger.info("Step 3: Deduplicating trends...")
        # deduplicated_trends = deduplicate_trends(trends_with_context, save_path="trends_deduped.json")
        # logger.info("Trends deduplicated.")

        # Final output
        logger.info("Pipeline completed successfully!")
        return trends_with_context

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    logger.info("Starting the pipeline...")
    final_output = run_pipeline()
    logger.info(f"Final output: {final_output}")
