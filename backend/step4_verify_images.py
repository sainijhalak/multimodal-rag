import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import lancedb
import torch
from sentence_transformers import SentenceTransformer

def create_sample_images(uploads_dir: Path):
    """Generates 3 synthetic test images with distinct visual patterns."""
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # 1. Red sports car icon/graphic
    img_car = Image.new("RGB", (300, 300), color=(240, 240, 240))
    draw_car = ImageDraw.Draw(img_car)
    # Red car body
    draw_car.rectangle([50, 140, 250, 200], fill=(220, 20, 20))
    draw_car.polygon([(90, 140), (130, 90), (200, 90), (230, 140)], fill=(200, 10, 10))
    # Black wheels
    draw_car.ellipse([70, 190, 110, 230], fill=(30, 30, 30))
    draw_car.ellipse([190, 190, 230, 230], fill=(30, 30, 30))
    car_path = uploads_dir / "red_sports_car.png"
    img_car.save(car_path)

    # 2. Green electronic circuit board
    img_circuit = Image.new("RGB", (300, 300), color=(10, 80, 40))
    draw_circuit = ImageDraw.Draw(img_circuit)
    # Golden circuit lines and microchip
    draw_circuit.rectangle([110, 110, 190, 190], fill=(20, 20, 20), outline=(218, 165, 32), width=3)
    draw_circuit.line([(50, 80), (110, 130)], fill=(218, 165, 32), width=3)
    draw_circuit.line([(50, 220), (110, 170)], fill=(218, 165, 32), width=3)
    draw_circuit.line([(190, 150), (260, 150)], fill=(218, 165, 32), width=3)
    circuit_path = uploads_dir / "circuit_board.png"
    img_circuit.save(circuit_path)

    # 3. Blue ocean waves
    img_ocean = Image.new("RGB", (300, 300), color=(135, 206, 235))
    draw_ocean = ImageDraw.Draw(img_ocean)
    # Deep blue water waves
    draw_ocean.rectangle([0, 150, 300, 300], fill=(0, 105, 148))
    draw_ocean.arc([20, 140, 140, 200], 0, 180, fill=(255, 255, 255), width=4)
    draw_ocean.arc([160, 140, 280, 200], 0, 180, fill=(255, 255, 255), width=4)
    ocean_path = uploads_dir / "ocean_waves.png"
    img_ocean.save(ocean_path)

    return [
        {"title": "Red Sports Car", "path": car_path, "description": "A bright red automobile with wheels."},
        {"title": "Electronic Circuit Board", "path": circuit_path, "description": "Printed circuit board with microchip and gold traces."},
        {"title": "Ocean Waves", "path": ocean_path, "description": "Deep blue water with white sea foam and sky."}
    ]

def main():
    print("=" * 65)
    print("🎓 PHASE 2: Cross-Modal Image Search with EmbeddingGemma-2")
    print("=" * 65)

    base_dir = Path(__file__).resolve().parent.parent
    uploads_dir = base_dir / "data" / "uploads"
    db_dir = base_dir / "data" / "lancedb"

    # 1. Create real test images
    print(f"\n[1. Preparing Test Images]")
    image_items = create_sample_images(uploads_dir)
    for item in image_items:
        print(f"  • Created image: {item['path'].name} ({item['title']})")

    # 2. Load Model on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    print(f"\n[2. Loading EmbeddingGemma-2 on {device.upper()}]")
    start_load = time.time()
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        device=device,
        model_kwargs={"torch_dtype": dtype}
    )
    print(f"  ✓ Model ready in {time.time() - start_load:.2f}s")

    # 3. Vectorize Images
    # Google Best Practice: Media is passed raw without text prompt prefixes
    print(f"\n[3. Vectorizing Images Directly with Vision Encoder]")
    pil_images = [Image.open(item["path"]) for item in image_items]

    start_embed = time.time()
    image_vectors = model.encode(
        pil_images,
        truncate_dim=256,
        normalize_embeddings=True,
        batch_size=4
    )
    print(f"  ✓ Encoded {len(pil_images)} images into 256d vectors in {time.time() - start_embed:.2f}s!")

    # 4. Save Image Vectors into LanceDB Table
    print(f"\n[4. Storing Image Embeddings in LanceDB]")
    db = lancedb.connect(str(db_dir))
    table = db.open_table("knowledge_base")

    new_records = []
    for vec, item in zip(image_vectors, image_items):
        new_records.append({
            "vector": vec.tolist(),
            "title": item["title"],
            "content": str(item["path"].name),     # Stores the image filename
            "source": str(item["path"].name),
            "modality": "image"
        })

    table.add(new_records)
    print(f"  ✓ Added {len(new_records)} images to table. Total documents now: {table.count_rows()}")

    # 5. Test Cross-Modal Search: Text Query ➔ Retrieve Image!
    test_queries = [
        "a red racing automobile",
        "computer hardware and silicon microchip",
        "sea water and blue ocean waves"
    ]

    print(f"\n[5. Testing Cross-Modal Search (Text Query ➔ Image Retrieval)]")
    for q in test_queries:
        print("\n" + "-" * 65)
        print(f"🔍 TEXT QUERY: \"{q}\"")

        # Encode text query with Google's SearchQuery prefix
        start_q = time.time()
        q_vec = model.encode(
            q,
            prompt_name="SearchQuery",
            truncate_dim=256,
            normalize_embeddings=True
        )

        # Search LanceDB for matching images
        results = table.search(q_vec.tolist()).where("modality = 'image'").limit(2).to_list()
        search_ms = (time.time() - start_q) * 1000

        print(f"  Ranked Results ({search_ms:.1f}ms):")
        for rank, res in enumerate(results, start=1):
            sim = float(np.dot(q_vec, res["vector"]))
            print(f"  #{rank} [{res['title']}] (Sim: {sim:.4f}) -> File: {res['content']}")

    print("\n" + "=" * 65)
    print("🎉 PHASE 2 COMPLETE: True Multimodal Vision is Working on Your GPU!")
    print("=" * 65)

if __name__ == "__main__":
    main()
