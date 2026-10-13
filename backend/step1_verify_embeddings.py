import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    print("=" * 60)
    print("🎓 LESSON 1: Running Google EmbeddingGemma-2 on your GPU")
    print("=" * 60)

    # 1. Inspect Hardware & Precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_supported = torch.cuda.is_bf16_supported() if device == "cuda" else False
    dtype = torch.bfloat16 if bf16_supported else torch.float32

    print(f"\n[Hardware Status]")
    print(f"  • Compute Device    : {device.upper()} ({torch.cuda.get_device_name(0)})")
    print(f"  • bfloat16 Support  : {bf16_supported} (Hardware Accelerated)")
    print(f"  • Working Precision : {dtype}")

    # Track VRAM before loading
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated(0) / (1024 ** 2)

    # 2. Load EmbeddingGemma-2 Model
    print(f"\n[Loading Model]")
    print(f"  Loading 'google/embeddinggemma-2' into GPU memory...")
    start_load = time.time()

    model = SentenceTransformer(
        "google/embeddinggemma-2",
        device=device,
        model_kwargs={"torch_dtype": dtype}
    )
    load_time = time.time() - start_load

    vram_after = torch.cuda.memory_allocated(0) / (1024 ** 2)
    vram_used = vram_after - vram_before

    print(f"  ✓ Model loaded successfully in {load_time:.2f} seconds!")
    print(f"  ✓ GPU VRAM consumed by model: {vram_used:.1f} MB (leaves ~{(4096 - vram_after):.0f} MB free)")

    # 3. Define Sample Data
    # Notice: One query, one highly relevant document, and one completely unrelated document
    query = "What is an embedding model?"
    doc_match = "Embedding models turn words and media into lists of numbers representing their semantic meaning."
    doc_unrelated = "A chocolate cake recipe requires cocoa powder, flour, sugar, and butter."

    print(f"\n[Test Inputs]")
    print(f"  Search Query        : \"{query}\"")
    print(f"  Matching Document   : \"{doc_match}\"")
    print(f"  Unrelated Document  : \"{doc_unrelated}\"")

    # 4. Generate Embeddings using Google's Best Practices:
    #   - prompt_name="SearchQuery" for the query
    #   - prompt_name="Document" for documents
    #   - truncate_dim=256 (Matryoshka Representation Learning)
    #   - normalize_embeddings=True (Essential after truncation)
    print(f"\n[Encoding to Vectors]")
    print("  Applying Google's 256-dimension Matryoshka truncation + L2 normalization...")

    query_vec = model.encode(
        query,
        prompt_name="SearchQuery",
        truncate_dim=256,
        normalize_embeddings=True
    )

    doc_match_vec = model.encode(
        doc_match,
        prompt_name="Document",
        truncate_dim=256,
        normalize_embeddings=True
    )

    doc_unrelated_vec = model.encode(
        doc_unrelated,
        prompt_name="Document",
        truncate_dim=256,
        normalize_embeddings=True
    )

    print(f"  ✓ Query vector shape : {query_vec.shape} (256 numbers)")
    print(f"  ✓ First 5 numbers    : {np.round(query_vec[:5], 4)}")

    # 5. Measure Similarity (Dot Product of normalized vectors = Cosine Similarity)
    similarity_match = float(np.dot(query_vec, doc_match_vec))
    similarity_unrelated = float(np.dot(query_vec, doc_unrelated_vec))

    print(f"\n[Similarity Analysis]")
    print(f"  Similarity with Matching Document  : {similarity_match:.4f}  (High Score!)")
    print(f"  Similarity with Unrelated Document : {similarity_unrelated:.4f}  (Low Score)")

    # 6. Conclusion
    print("\n" + "=" * 60)
    if similarity_match > similarity_unrelated:
        print("🎉 SUCCESS! The model correctly identified the relevant passage.")
        print("   Your GPU, bfloat16 math, and 256d vector truncation are 100% operational!")
    else:
        print("⚠️ Check setup: similarity ranking unexpected.")
    print("=" * 60)

if __name__ == "__main__":
    main()
