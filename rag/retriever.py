
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH      = "rag/faiss_index.bin"
CHUNKS_PATH     = "rag/chunks.pkl"
TOP_K_DEFAULT   = 3

# ── Model (loaded once, reused across calls) ─────────────────────────────────

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[retriever] Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ── Index Building ────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[str]) -> faiss.IndexFlatL2:
    model = _get_model()

    print(f"[retriever] Encoding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # L2 distance index — dimension = embedding size (384 for MiniLM)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # Persist to disk so we don't rebuild every run
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[retriever] Index built: {index.ntotal} vectors, dim={dimension}")
    return index


def load_faiss_index() -> tuple[faiss.IndexFlatL2, list[str]]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "FAISS index not found. Run `indexer.build_and_save_index()` first."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"[retriever] Loaded index: {index.ntotal} vectors")
    return index, chunks


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    k: int = TOP_K_DEFAULT,
) -> str:
    if index.ntotal == 0:
        return ""

    model = _get_model()
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # FAISS returns distances and indices of nearest neighbours
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            chunk_text = chunks[idx].strip()
            retrieved_chunks.append(f"[Source {i+1}]\n{chunk_text}")

    return "\n\n".join(retrieved_chunks)


# ── Quick Smoke Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build a tiny index and test retrieval
    sample_chunks = [
        "Traffic congestion in Indian cities is caused by mixed traffic, encroachments, and poor signal timing.",
        "BRT (Bus Rapid Transit) systems have reduced urban travel times by 30% in cities like Ahmedabad.",
        "Plastic waste in Indian colleges generates around 500kg of single-use plastic per year per institution.",
        "Composting organic waste at source reduces landfill load by up to 60%.",
        "Rainwater harvesting can replenish 25-40% of a household's annual water needs.",
    ]

    print("--- Building index ---")
    idx = build_faiss_index(sample_chunks)

    print("\n--- Testing retrieval ---")
    query = "How do I reduce traffic in a city?"
    result = retrieve(query, idx, sample_chunks, k=2)
    print(f"Query: {query}\n\nRetrieved:\n{result}")
