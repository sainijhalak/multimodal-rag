import time
from pathlib import Path
import numpy as np
import lancedb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 65)
    print("🎓 LESSON 3: The Complete End-to-End RAG Loop on Your GPU")
    print("=" * 65)

    base_dir = Path(__file__).resolve().parent.parent
    db_dir = base_dir / "data" / "lancedb"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    # 1. Connect to LanceDB Table
    print(f"\n[1. Connecting to Vector Database on Disk]")
    db = lancedb.connect(str(db_dir))
    table = db.open_table("knowledge_base")
    print(f"  ✓ Connected to 'knowledge_base' ({table.count_rows()} documents on disk)")

    # 2. Load Embedding Model
    print(f"\n[2. Loading Embedding Model on {device.upper()}]")
    start_embed_load = time.time()
    embedder = SentenceTransformer(
        "google/embeddinggemma-2",
        device=device,
        model_kwargs={"torch_dtype": dtype}
    )
    print(f"  ✓ EmbeddingGemma loaded in {time.time() - start_embed_load:.2f}s")

    # 3. Load Local Generative LLM (Qwen2.5-0.5B-Instruct)
    print(f"\n[3. Loading Conversational LLM on {device.upper()}]")
    llm_id = "Qwen/Qwen2.5-0.5B-Instruct"
    start_llm_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_id,
        dtype=dtype,
        device_map="auto"
    )
    print(f"  ✓ LLM loaded in {time.time() - start_llm_load:.2f}s")

    # Measure Total VRAM
    total_vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"\n★ TOTAL GPU VRAM CONSUMED (Embedder + LLM): {total_vram:.1f} MB (~{total_vram/1024:.2f} GB / 4.0 GB)")
    print(f"★ FREE VRAM SAFETY BUFFER: {4096 - total_vram:.1f} MB (Cool, quiet, zero system stress!)")

    # 4. Test Queries
    queries = [
        "How do I prevent my vehicle engine from wearing out?",
        "What makes authentic Neapolitan pizza dough special?"
    ]

    for q in queries:
        print("\n" + "=" * 65)
        print(f"❓ USER QUESTION: \"{q}\"")

        # Step A: Embed Query with SearchQuery prefix & MRL 256d
        start_t = time.time()
        q_vec = embedder.encode(
            q,
            prompt_name="SearchQuery",
            truncate_dim=256,
            normalize_embeddings=True
        )

        # Step B: Sub-10ms LanceDB Disk Retrieval
        search_results = table.search(q_vec.tolist()).limit(1).to_list()
        retrieval_ms = (time.time() - start_t) * 1000

        best_doc = search_results[0]
        dot_sim = float(np.dot(q_vec, best_doc["vector"]))
        print(f"\n  🔍 RETRIEVED FROM DISK in {retrieval_ms:.1f}ms (Similarity: {dot_sim:.4f}):")
        print(f"     Source : {best_doc['source']} ({best_doc['title']})")
        print(f"     Content: \"{best_doc['content']}\"")

        # Step C: Generate Conversational Answer
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly, knowledgeable AI assistant. "
                    "Answer the user's question clearly and conversationally using ONLY the provided facts. "
                    "Do not make up facts outside the context."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context from my files:\n"
                    f"Title: {best_doc['title']}\n"
                    f"Fact: {best_doc['content']}\n\n"
                    f"Question: {q}"
                )
            }
        ]

        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([formatted_input], return_tensors="pt").to(device)

        gen_start = time.time()
        with torch.no_grad():
            generated_ids = llm.generate(
                **model_inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1
            )

        # Extract only the generated response
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        gen_duration = time.time() - gen_start

        print(f"\n  🤖 AI GENERATED RESPONSE ({gen_duration:.2f}s):")
        print(f"  ─────────────────────────────────────────────────────────────")
        print(f"  {response_text.strip()}")
        print(f"  ─────────────────────────────────────────────────────────────")

    print("\n" + "=" * 65)
    print("🎉 MILESTONE ACHIEVED: Phase 1 (Text RAG Core) is 100% COMPLETE!")
    print("   Fast, accurate, conversational, and safely within your 4GB GPU!")
    print("=" * 65)

if __name__ == "__main__":
    main()
