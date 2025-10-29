from langchain_text_splitters import RecursiveCharacterTextSplitter


def split(docs):
  """Split one or many docs into chunked records.

  docs can be a single dict or a list of dicts.
  Each input doc requires keys: id, title, content, metadata.
  """
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=100,
  )

  # Normalize input
  items = docs if isinstance(docs, (list, tuple)) else [docs]

  records = []
  for doc in items:
      chunks = splitter.split_text(doc["content"])
      for i, chunk in enumerate(chunks):
          records.append({
              "id": f"{doc['id']}:{i}",
              "content": f"{doc['title']}\n\n{chunk}",
              "metadata": {**doc["metadata"], "source_id": doc["id"], "chunk": i},
          })

  return records
