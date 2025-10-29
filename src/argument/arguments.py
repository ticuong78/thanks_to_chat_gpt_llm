from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from .constants import EMBEDDING

host = "http://localhost:11434"
data_path = "./src/data/posts_content.jsonl"

# Embedding function backed by local Ollama
embeddings = OllamaEmbeddings(model=EMBEDDING, base_url=host)

# Persistent Chroma DB configured with embedding function
db = Chroma(
    collection_name="posts",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

__all__ = ("host", "data_path", "db", "embeddings")
