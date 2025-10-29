import os


def env(key: str, default: str) -> str:
  return os.environ.get(key, default)


# Core settings (override via env if needed)
HOST = env("LLM_HOST", "http://localhost:11434")
EMBEDDING_MODEL = env("EMBEDDING_MODEL", "nomic-embed-text")
GENERATION_MODEL = env("GENERATION_MODEL", "llama3.2:latest")

DATA_PATH = env("DATA_PATH", "./src/data/posts_content.jsonl")
PERSIST_DIR = env("PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = env("COLLECTION_NAME", "posts")


__all__ = (
  "HOST",
  "EMBEDDING_MODEL",
  "GENERATION_MODEL",
  "DATA_PATH",
  "PERSIST_DIR",
  "COLLECTION_NAME",
)

