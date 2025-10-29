from ..singletons import ollama_client

def get_embedding(text: str, model: str = "nomic-embed-text"):
  response = ollama_client.embeddings(model=model, prompt=text)
  return response["embedding"]