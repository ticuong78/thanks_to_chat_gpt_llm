import os
import json
import hashlib
import shutil
from loguru import logger
from tqdm import tqdm
from .argument.arguments import data_path
from .reading.json_read import json_read
from .embedding import embed


FINGERPRINT_PATH = "./chroma_db/fingerprint.json"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fingerprint() -> dict | None:
    try:
        with open(FINGERPRINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_fingerprint(fp: dict) -> None:
    os.makedirs(os.path.dirname(FINGERPRINT_PATH), exist_ok=True)
    with open(FINGERPRINT_PATH, "w", encoding="utf-8") as f:
        json.dump(fp, f, ensure_ascii=False, indent=2)


def needs_rebuild() -> bool:
    current = {
        "data_path": data_path,
        "sha256": file_sha256(data_path),
    }
    previous = load_fingerprint()
    if previous == current and os.path.exists("./chroma_db"):
        logger.info("No data changes detected. Skipping rebuild.")
        return False
    logger.info("Data changed or DB missing. Will (re)build vector store.")
    return True


def rebuild_chroma(docs: list[dict]) -> None:
    # Fresh rebuild: remove on-disk DB (simplest/most reliable)
    if os.path.exists("./chroma_db"):
        logger.info("Removing existing Chroma DB directory...")
        shutil.rmtree("./chroma_db", ignore_errors=True)

    # Add all docs in chunks
    logger.info("Start embedding and adding to RAG...")
    for doc in tqdm(docs, desc="Embedding docs"):
        embed(doc)

    # Write fingerprint to mark successful build
    save_fingerprint({"data_path": data_path, "sha256": file_sha256(data_path)})
    logger.success("Successfully (re)created RAG.")


def main():
    try:
        logger.info(f"Reading texts from {data_path}...")
        docs = json_read(data_path)

        if needs_rebuild():
            rebuild_chroma(docs)
        else:
            logger.info("Using existing vector store at ./chroma_db")

    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
