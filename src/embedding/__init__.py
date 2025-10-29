from .splitting import split
from ..argument.arguments import db
from langchain_core.documents import Document
from loguru import logger


def embed(docs) -> None:
  """Chunk docs and add to Chroma with stable IDs.

  Accepts a single doc or a list of docs.
  """
  try:
    logger.info("Splitting docs...")
    records = split(docs)

    logger.info("Adding to vector store...")
    documents = []
    ids = []
    for r in records:
      documents.append(Document(page_content=r["content"], metadata=r["metadata"]))
      ids.append(r["id"])

    if documents:
      db.add_documents(documents=documents, ids=ids)

    logger.success(f"Added {len(documents)} chunks to RAG")

  except Exception as e:
    logger.error(e)

__all__ = ("embed",)
