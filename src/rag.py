"""
rag.py — ChromaDB-backed RAG pipeline with sentence-transformer embeddings.

CLI:
    python src/rag.py --ingest data/documents/
    python src/rag.py --query "what is retrieval augmented generation?"
    python src/rag.py --ingest data/documents/ --query "transformer attention"
"""

import argparse
import re
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "multimodal_rag"
CHUNK_SIZE = 500        # characters
CHUNK_OVERLAP = 50      # characters
TOP_K = 3

# ---------------------------------------------------------------------------
# Singletons (lazy-loaded once per process)
# ---------------------------------------------------------------------------

_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model '{EMBED_MODEL_NAME}' …")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into chunks of ~chunk_size characters with *overlap* character
    overlap. Chunk boundaries snap to the nearest sentence end ('. ', '? ', '! ')
    when one exists in the upper half of the window.
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    # Pre-compute all sentence-end positions (position just after the space)
    boundary_pattern = re.compile(r"(?<=[.?!])\s+")
    boundaries = [m.end() for m in boundary_pattern.finditer(text)]

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap          # guaranteed positive (500-50=450)

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Snap to a sentence boundary in the upper half of the window
        if end < len(text):
            lo, hi = start + chunk_size // 2, end
            candidates = [b for b in boundaries if lo < b <= hi]
            if candidates:
                end = max(candidates)    # latest boundary within the window

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance by step; on the last chunk stop to avoid an empty tail
        if end >= len(text):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_documents(directory: str) -> int:
    """
    Load all .txt files from *directory*, chunk them, and upsert into ChromaDB.
    Returns the total number of chunks ingested.
    """
    doc_dir = Path(directory)
    txt_files = sorted(doc_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in '{directory}'.")
        return 0

    embedder = _get_embedder()
    collection = _get_collection()
    total_chunks = 0

    for txt_path in tqdm(txt_files, desc="Ingesting files"):
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        if not chunks:
            continue

        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
        metadatas = [{"source": txt_path.name, "chunk_index": i} for i, _ in enumerate(chunks)]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        tqdm.write(f"  {txt_path.name}: {len(chunks)} chunks")

    print(f"\nIngestion complete — {total_chunks} chunks stored in '{CHROMA_DIR}/'.")
    return total_chunks


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed *query* and return the top-k most similar chunks from ChromaDB.

    Each result dict has keys: text, source, chunk_index, distance.
    """
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": dist,
            }
        )

    return hits


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_context(hits: list[dict]) -> str:
    """
    Format retrieval results as labelled source blocks suitable for an LLM prompt.
    """
    if not hits:
        return "No relevant context found."

    blocks = []
    for i, hit in enumerate(hits, 1):
        similarity = 1.0 - hit["distance"]   # cosine distance → similarity
        header = f"[Source {i}] {hit['source']} (chunk {hit['chunk_index']}, similarity {similarity:.3f})"
        blocks.append(f"{header}\n{hit['text']}")

    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal RAG — ingest documents and query with ChromaDB"
    )
    parser.add_argument("--ingest", metavar="DIR", help="Directory of .txt files to ingest")
    parser.add_argument("--query", metavar="TEXT", help="Query to retrieve relevant chunks for")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Number of results (default {TOP_K})")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.ingest and not args.query:
        parser.print_help()
        return

    if args.ingest:
        ingest_documents(args.ingest)

    if args.query:
        print(f"\nQuery: {args.query!r}\n")
        hits = retrieve(args.query, top_k=args.top_k)
        print(format_context(hits))


if __name__ == "__main__":
    main()
