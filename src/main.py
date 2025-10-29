from loguru import logger
from .app.config import DATA_PATH
from .app.vectorstore.chroma_store import VectorStoreManager


def main():
  try:
    manager = VectorStoreManager()
    if manager.needs_rebuild(DATA_PATH):
      logger.info("Data changed or DB missing. Rebuilding vector store...")
      manager.rebuild_from_jsonl(DATA_PATH)
      logger.success("Vector store (re)built successfully.")
    else:
      logger.info("No data changes detected. Using existing vector store.")
  except Exception as e:
    logger.exception(e)


if __name__ == "__main__":
  main()
