import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image, ImageDraw
import lancedb
import torch
from sentence_transformers import SentenceTransformer

def create_sample_video(video_path: Path):
    """
    Creates a 6-second synthetic MP4 video with 3 distinct visual sections:
      - Seconds 0-2: Red sports car on asphalt track
      - Seconds 2-4: Electronic circuit board with glowing microchips
      - Seconds 4-6: Blue ocean water waves
    """
    import av
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fps = 10
    duration = 6
    total_frames = fps * duration

    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"

    for i in range(total_frames):
        sec = i / fps
        img = Image.new("RGB", (320, 240))
        draw = ImageDraw.Draw(img)

        if sec < 2.0:
            # Red sports car on asphalt
            draw.rectangle([(0, 0), (320, 140)], fill=(135, 206, 235))  # Sky
            draw.rectangle([(0, 140), (320, 240)], fill=(50, 50, 50))   # Asphalt Road
            # Moving car
            car_x = int(30 + (sec / 2.0) * 160)
            draw.rectangle([(car_x, 150), (car_x + 90, 180)], fill=(220, 20, 60)) # Red car body
            draw.ellipse([(car_x + 10, 175), (car_x + 25, 190)], fill=(10, 10, 10)) # Wheel
            draw.ellipse([(car_x + 65, 175), (car_x + 80, 190)], fill=(10, 10, 10)) # Wheel
            draw.text((10, 10), f"SEC {sec:.1f}: High-speed red vehicle", fill=(0, 0, 0))

        elif sec < 4.0:
            # Green circuit board with microchips
            draw.rectangle([(0, 0), (320, 240)], fill=(15, 80, 35)) # Green PCB
            # Microchip
            draw.rectangle([(110, 70), (210, 170)], fill=(20, 20, 20)) # Silicon chip
            draw.rectangle([(120, 80), (200, 160)], fill=(40, 40, 40))
            # Gold circuit traces
            draw.line([(0, 60), (110, 90)], fill=(218, 165, 32), width=3)
            draw.line([(0, 180), (110, 150)], fill=(218, 165, 32), width=3)
            draw.line([(210, 90), (320, 50)], fill=(218, 165, 32), width=3)
            draw.line([(210, 150), (320, 200)], fill=(218, 165, 32), width=3)
            draw.text((10, 10), f"SEC {sec:.1f}: Microcontroller circuit board", fill=(255, 255, 255))

        else:
            # Blue ocean waves
            draw.rectangle([(0, 0), (320, 240)], fill=(0, 105, 148)) # Ocean blue
            wave_offset = int((sec - 4.0) * 40)
            draw.arc([(20 - wave_offset, 60), (180 - wave_offset, 140)], start=0, end=180, fill=(240, 255, 255), width=4)
            draw.arc([(140 - wave_offset, 120), (300 - wave_offset, 200)], start=0, end=180, fill=(240, 255, 255), width=4)
            draw.arc([(60 + wave_offset, 160), (220 + wave_offset, 230)], start=0, end=180, fill=(255, 255, 255), width=3)
            draw.text((10, 10), f"SEC {sec:.1f}: Deep blue ocean water surf", fill=(255, 255, 255))

        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return video_path

def main():
    print("=" * 65)
    print("🎓 PHASE 4: Multimodal Video (Seeing Over Time) with EmbeddingGemma-2")
    print("=" * 65)

    base_dir = Path(__file__).resolve().parent.parent
    uploads_dir = base_dir / "data" / "uploads"
    db_dir = base_dir / "data" / "lancedb"
    video_path = uploads_dir / "tech_presentation.mp4"

    # 1. Synthesize 6-second test video
    print(f"\n[1. Synthesizing 6-Second MP4 Multimodal Video Clip]")
    create_sample_video(video_path)
    print(f"  ✓ Created video: {video_path.name} (320x240, 6.0s duration)")

    # 2. Extract Keyframes at 1 FPS
    print(f"\n[2. Extracting Keyframes at 1 FPS using VideoProcessor]")
    from app.processors.video_processor import VideoProcessor
    processor = VideoProcessor(target_fps=1)
    chunks = processor.process(video_path)
    for c in chunks:
        meta = c.metadata
        print(f"  • Keyframe @ {meta['timestamp_formatted']} ({meta['timestamp_sec']}s) -> {Path(c.content).name}")

    # 3. Load Embedding Model on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    print(f"\n[3. Loading EmbeddingGemma-2 Vision Encoder on {device.upper()}]")
    start_load = time.time()
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        device=device,
        model_kwargs={"torch_dtype": dtype}
    )
    print(f"  ✓ Model ready in {time.time() - start_load:.2f}s")

    # 4. Embed Video Keyframe Images
    print(f"\n[4. Vectorizing Keyframe Images into 256d Embeddings]")
    frame_images = [Image.open(c.content) for c in chunks]
    start_embed = time.time()
    frame_vectors = model.encode(
        frame_images,
        truncate_dim=256,
        normalize_embeddings=True,
        batch_size=4
    )
    print(f"  ✓ Encoded {len(frame_images)} keyframes in {time.time() - start_embed:.2f}s!")

    # 5. Store in LanceDB
    print(f"\n[5. Storing Video Frames in LanceDB with Temporal Timestamps]")
    db = lancedb.connect(str(db_dir))
    table = db.open_table("knowledge_base")

    new_records = []
    for vec, chunk in zip(frame_vectors, chunks):
        meta = chunk.metadata
        new_records.append({
            "vector": vec.tolist(),
            "title": f"Video Clip at {meta['timestamp_formatted']}",
            "content": f"{meta['source_video']} @ {meta['timestamp_formatted']}",
            "source": meta["source_video"],
            "modality": "video"
        })

    table.add(new_records)
    print(f"  ✓ Added {len(new_records)} video keyframe records. Total documents in DB now: {table.count_rows()}")

    # 6. Test Cross-Modal Video Retrieval (Text Query ➔ Exact Timestamp Seeking)
    test_queries = [
        "red sports car speeding on the highway",
        "silicon microchip and printed circuit board electronic traces",
        "deep blue ocean water waves and natural surf"
    ]

    print(f"\n[6. Testing Cross-Modal Video Search (Text Query ➔ Timestamp Retrieval)]")
    for q in test_queries:
        print("\n" + "-" * 65)
        print(f"🔍 TEXT QUERY: \"{q}\"")

        start_q = time.time()
        q_vec = model.encode(
            q,
            prompt_name="SearchQuery",
            truncate_dim=256,
            normalize_embeddings=True
        )

        results = table.search(q_vec.tolist()).where("modality = 'video'").limit(2).to_list()
        search_ms = (time.time() - start_q) * 1000

        print(f"  Ranked Results ({search_ms:.1f}ms):")
        for rank, res in enumerate(results, start=1):
            sim = float(np.dot(q_vec, res["vector"]))
            print(f"  #{rank} [{res['title']}] (Sim: {sim:.4f}) -> Content: {res['content']}")

    print("\n" + "=" * 65)
    print("🎉 PHASE 4 COMPLETE: Multimodal Video (Seeing Over Time) is Operational!")
    print("=" * 65)

if __name__ == "__main__":
    main()
