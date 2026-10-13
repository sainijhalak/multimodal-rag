import time
from pathlib import Path
import numpy as np
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import torch

def main():
    print("=" * 60)
    print("🎓 LESSON 2: Persistent Vector Storage with LanceDB")
    print("=" * 60)

    # 1. Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    db_dir = base_dir / "data" / "lancedb"
    uploads_dir = base_dir / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create sample knowledge files (simulating user uploads)
    documents = [
        {
            "title": "Quantum Computing Basics",
            "content": "Quantum computers use qubits which can exist in superposition states, solving specific problems exponentially faster than classical bits.",
            "source": "quantum_notes.txt"
        },
        {
            "title": "Car Engine Maintenance",
            "content": "Regular oil changes every 5,000 miles protect internal engine bearings and pistons from excessive friction and thermal breakdown.",
            "source": "car_manual.txt"
        },
        {
            "title": "Embedding Models Overview",
            "content": "Google EmbeddingGemma-2 maps text, images, and audio into a unified 256-dimensional vector space for semantic search and RAG.",
            "source": "ai_research.txt"
        },
        {
            "title": "Italian Pizza Dough Recipe",
            "content": "Authentic Neapolitan pizza dough uses high-protein double zero flour, 65 percent hydration, yeast, and a 24-hour cold fermentation.",
            "source": "recipes.txt"
        },
        {
            "title": "NVIDIA GPU Architecture",
            "content": "Modern NVIDIA GPUs like the RTX 3050 feature Ampere architecture tensor cores with native support for bfloat16 mixed precision math.",
            "source": "hardware_specs.txt"
        }
    ]

    print(f"\n[1. Ingesting Sample Documents]")
    for doc in documents:
        file_path = uploads_dir / doc["source"]
        file_path.write_text(f"title: {doc['title']} | text: {doc['content']}", encoding="utf-8")
        print(f"  • Created source file: {doc['source']} ({doc['title']})")

    # 3. Load Embedding Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f"\n[2. Loading Model on {device.upper()}]")
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        device=device,
        model_kwargs={"torch_dtype": dtype}
    )

    # 4. Generate Embeddings for all documents (Batched)
    print(f"\n[3. Vectorizing Documents]")
    corpus_texts = [f"title: {d['title']} | text: {d['content']}" for d in documents]
    
    start_embed = time.time()
    corpus_vectors = model.encode(
        corpus_texts,
        prompt_name="Document",
        truncate_dim=256,
        normalize_embeddings=True,
        batch_size=8
    )
    embed_duration = time.time() - start_embed
    print(f"  ✓ Encoded {len(documents)} documents in {embed_duration:.2f} seconds!")
    print(f"  ✓ Output matrix shape: {corpus_vectors.shape} (5 rows, 256 dimensions)")

    # 5. Initialize LanceDB on Disk
    print(f"\n[4. Initializing LanceDB Database on Disk]")
    print(f"  Database directory: {db_dir}")
    db = lancedb.connect(str(db_dir))

    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 256)),
        pa.field("title", pa.string()),
        pa.field("content", pa.string()),
        pa.field("source", pa.string()),
        pa.field("modality", pa.string()),
    ])

    table_name = "knowledge_base"
    # Overwrite if exists for clean testing
    table = db.create_table(table_name, schema=schema, mode="overwrite")

    # Add records to LanceDB
    records = []
    for vec, doc in zip(corpus_vectors, documents):
        records.append({
            "vector": vec.tolist(),
            "title": doc["title"],
            "content": doc["content"],
            "source": doc["source"],
            "modality": "text"
        })
    table.add(records)
    print(f"  ✓ Stored {len(records)} records in LanceDB table '{table_name}'")

    # 6. Test Real Semantic Queries
    test_queries = [
        "How do I prevent my vehicle engine from wearing out?",
        "What hardware can run bfloat16 fast?",
        "How do we represent multimodal inputs as vectors?"
    ]

    print(f"\n[5. Testing Vector Retrieval from Disk]")
    for q in test_queries:
        print(f"\n  🔍 User Query: \"{q}\"")
        q_vec = model.encode(
            q,
            prompt_name="SearchQuery",
            truncate_dim=256,
            normalize_embeddings=True
        )

        # Query LanceDB
        start_search = time.time()
        results = table.search(q_vec.tolist()).limit(2).to_list()
        search_duration = (time.time() - start_search) * 1000

        print(f"     Search completed in {search_duration:.2f} ms:")
        for rank, res in enumerate(results, start=1):
            # In LanceDB cosine search, _distance is 1 - cosine_similarity (for normalized vectors: distance = 1 - sim)
            # Similarity = 1 - distance / 2 or dot product
            dot_sim = float(np.dot(q_vec, res["vector"]))
            print(f"     #{rank} [{res['title']}] (Sim: {dot_sim:.4f}) -> Source: {res['source']}")
            print(f"        \"{res['content'][:70]}...\"")

    print("\n" + "=" * 60)
    print("🎉 SUCCESS: LanceDB vector database is fully operational on disk!")
    print("=" * 60)

if __name__ == "__main__":
    main()
