import json
import os
import shutil
import hashlib
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from ...reading.json_read import json_read
from ...embedding.splitting import split
from ..config import HOST, EMBEDDING_MODEL, PERSIST_DIR, COLLECTION_NAME


FINGERPRINT_FILE = "fingerprint.json"


def _sha256_file(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


class VectorStoreManager:
  def __init__(self,
               persist_directory: str = PERSIST_DIR,
               collection_name: str = COLLECTION_NAME,
               host: str = HOST,
               embedding_model: str = EMBEDDING_MODEL):
    self.persist_directory = persist_directory
    self.collection_name = collection_name
    self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=host)

  def open(self) -> Chroma:
    os.makedirs(self.persist_directory, exist_ok=True)
    return Chroma(
      collection_name=self.collection_name,
      persist_directory=self.persist_directory,
      embedding_function=self.embeddings,
    )

  # Fingerprint helpers
  def _fp_path(self) -> str:
    return os.path.join(self.persist_directory, FINGERPRINT_FILE)

  def _load_fp(self) -> Dict[str, Any] | None:
    try:
      with open(self._fp_path(), "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      return None

  def _save_fp(self, fp: Dict[str, Any]) -> None:
    os.makedirs(self.persist_directory, exist_ok=True)
    with open(self._fp_path(), "w", encoding="utf-8") as f:
      json.dump(fp, f, ensure_ascii=False, indent=2)

  def needs_rebuild(self, data_path: str) -> bool:
    current = {"data_path": data_path, "sha256": _sha256_file(data_path)}
    previous = self._load_fp()
    if previous == current and os.path.exists(self.persist_directory):
      return False
    return True

  def rebuild_from_docs(self, docs: List[Dict[str, Any]]) -> None:
    # Start fresh for correctness
    if os.path.exists(self.persist_directory):
      shutil.rmtree(self.persist_directory, ignore_errors=True)
    db = self.open()

    # Chunk and add with stable IDs
    for doc in docs:
      records = split(doc)
      if not records:
        continue
      documents = [Document(page_content=r["content"], metadata=r["metadata"]) for r in records]
      ids = [r["id"] for r in records]
      db.add_documents(documents=documents, ids=ids)

  def rebuild_from_jsonl(self, data_path: str) -> None:
    docs = json_read(data_path)
    self.rebuild_from_docs(docs)
    self._save_fp({"data_path": data_path, "sha256": _sha256_file(data_path)})

