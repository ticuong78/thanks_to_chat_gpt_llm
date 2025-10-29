import json


def json_read(path: str) -> list:
  docs = []
  
  with open(path, "r", encoding="utf-8") as f:
      for line in f:
          doc = json.loads(line)
          docs.append(doc)

  return docs

__all__ = ("json_read",)