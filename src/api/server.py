from flask import Flask, jsonify, request
from flask_cors import CORS
from time import perf_counter

from ..app.vectorstore.chroma_store import VectorStoreManager
from ..app.config import GENERATION_MODEL
from ..singletons import ollama_client


app = Flask(__name__)
CORS(app)
manager = VectorStoreManager()
db = manager.open()


def _format_sources(hits):
  sources = []
  for doc, score in hits:
    meta = dict(doc.metadata or {})
    sources.append({
      "id": meta.get("source_id"),
      "chunk": meta.get("chunk"),
      "score": score,
      "metadata": meta,
    })
  return sources


def _build_context(hits, max_chars: int = 3500):
  parts = []
  total = 0
  for doc, _ in hits:
    txt = doc.page_content.strip()
    if not txt:
      continue
    if total + len(txt) > max_chars:
      remaining = max_chars - total
      if remaining > 0:
        parts.append(txt[:remaining])
        total += remaining
      break
    parts.append(txt)
    total += len(txt)
  return "\n\n---\n\n".join(parts)


def _prompt_for_style(style: str, language: str, query: str, context: str):
  if language.lower().startswith("vi"):
    sys = (
      "Bạn là trợ lý trả lời ngắn gọn, chính xác. "
      "Chỉ sử dụng thông tin trong CONTEXT để trả lời. "
      "Nếu không đủ thông tin, hãy trả lời: 'Tôi không chắc từ dữ liệu đã lập chỉ mục.'"
    )
    if style == "structured":
      user = f"CÂU HỎI: {query}\n\nCONTEXT:\n{context}\n\nTRẢ LỜI có cấu trúc: \n- Tóm tắt\n- Điểm chính\n- Trích dẫn ngắn\n- Nguồn (source_id:chunk)"
    elif style == "qa":
      user = f"CÂU HỎI: {query}\n\nCONTEXT:\n{context}\n\nHãy suy nghĩ có hệ thống: liệt kê dữ kiện từ CONTEXT, sau đó trả lời ngắn gọn."
    else:
      user = f"CÂU HỎI: {query}\n\nCONTEXT:\n{context}\n\nTrả lời ngắn gọn, ưu tiên gạch đầu dòng."
  else:
    sys = (
      "You are a concise, accurate assistant. "
      "Use only the provided CONTEXT. "
      "If insufficient, answer: 'I am not sure from the indexed data.'"
    )
    if style == "structured":
      user = f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nAnswer with sections: \n- Summary\n- Key points\n- Short quotes\n- Sources (source_id:chunk)"
    elif style == "qa":
      user = f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nThink systematically: list facts from CONTEXT, then answer briefly."
    else:
      user = f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nAnswer briefly, prefer bullet points."
  return sys, user


@app.get("/health")
def health():
  try:
    _ = db._collection.count()  # type: ignore[attr-defined]
    return jsonify({"status": "ok", "docs": int(_)}), 200
  except Exception:
    return jsonify({"status": "ok"}), 200


@app.post("/query")
def query():
  t0 = perf_counter()
  body = request.get_json(silent=True) or {}
  q = body.get("query") or body.get("q")
  k = int(body.get("k", 6))
  threshold = body.get("score_threshold")
  style = str(body.get("prompt_style", "concise")).lower()
  language = str(body.get("language", "vi"))
  mmr = bool(body.get("mmr", False))
  fetch_k = int(body.get("fetch_k", max(8, k * 2)))

  if not q:
    return jsonify({"error": "Missing 'query' in JSON body"}), 400

  try:
    if mmr:
      docs = db.max_marginal_relevance_search(q, k=k, fetch_k=fetch_k)
      hits = [(d, 0.0) for d in docs]
    else:
      hits = db.similarity_search_with_score(q, k=k)

    if threshold is not None and not mmr:
      try:
        thr = float(threshold)
        hits = [(d, s) for (d, s) in hits if s <= thr]
      except Exception:
        pass

    if not hits:
      return jsonify({
        "query": q,
        "answer": "Tôi không chắc từ dữ liệu đã lập chỉ mục." if language.startswith("vi") else "I am not sure from the indexed data.",
        "sources": [],
        "latency_ms": int((perf_counter() - t0) * 1000),
      }), 200

    context = _build_context(hits)
    sys, user = _prompt_for_style(style, language, q, context)

    resp = ollama_client.chat(
      model=GENERATION_MODEL,
      messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
      options={"temperature": 0.2},
    )
    answer = resp.get("message", {}).get("content", "")

    return jsonify({
      "query": q,
      "answer": answer,
      "sources": _format_sources(hits),
      "used_prompt": {"system": sys, "user": user},
      "mmr": mmr,
      "k": k,
      "latency_ms": int((perf_counter() - t0) * 1000),
    }), 200
  except Exception as e:
    return jsonify({"error": str(e)}), 500


def create_app():
  return app


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8000, debug=False)
