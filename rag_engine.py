from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


DOCUMENTS_DIR = Path("documents")


def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def split_text(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def build_index():

    documents = load_documents()

    chunks = []
    metadata = []

    for document in documents:

        document_chunks = split_text(document["text"])

        for chunk in document_chunks:

            chunks.append(chunk)

            metadata.append({
                "source": document["source"],
                "text": chunk
            })

    # Load embedding model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    # FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.asarray(embeddings, dtype="float32")
    )

    return model, index, metadata


def search(query, model, index, metadata, top_k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        np.asarray(query_embedding, dtype="float32"),
        top_k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        results.append({
            "source": metadata[idx]["source"],
            "text": metadata[idx]["text"],
            "distance": float(distance)
        })

    return results