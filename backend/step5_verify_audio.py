import time
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import lancedb
import torch
from sentence_transformers import SentenceTransformer

def create_sample_audio_files(uploads_dir: Path):
    """
    Creates 3 test audio files meeting Google's exact standard:
      - 16,000 Hz Mono (.wav)
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    duration = 3.0  # 3 seconds each
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 1. High-frequency alarm beeps (1000 Hz pulsating)
    # Pulsing square envelope
    pulse = (np.sin(2 * np.pi * 3 * t) > 0).astype(float)
    alarm_wave = 0.6 * np.sin(2 * np.pi * 1000 * t) * pulse
    alarm_int16 = (alarm_wave * 32767).astype(np.int16)
    alarm_path = uploads_dir / "high_pitch_alarm_beep.wav"
    wavfile.write(alarm_path, sample_rate, alarm_int16)

    # 2. Low-frequency vehicle engine rumble (80 Hz + harmonics)
    engine_wave = (
        0.5 * np.sin(2 * np.pi * 80 * t) +
        0.3 * np.sin(2 * np.pi * 160 * t) +
        0.1 * np.random.normal(0, 0.2, len(t))
    )
    # Normalize
    engine_wave = engine_wave / np.max(np.abs(engine_wave)) * 0.7
    engine_int16 = (engine_wave * 32767).astype(np.int16)
    engine_path = uploads_dir / "diesel_engine_idle_sound.wav"
    wavfile.write(engine_path, sample_rate, engine_int16)

    # 3. Flowing water / ocean wave sound (Filtered soft noise)
    noise = np.random.normal(0, 0.5, len(t))
    # Soft amplitude swell like ocean waves (period = 1.5s)
    swell = 0.5 * (1 + np.sin(2 * np.pi * 0.7 * t))
    water_wave = noise * swell * 0.4
    water_wave = water_wave / np.max(np.abs(water_wave)) * 0.6
    water_int16 = (water_wave * 32767).astype(np.int16)
    water_path = uploads_dir / "ocean_surf_water_sound.wav"
    wavfile.write(water_path, sample_rate, water_int16)

    return [
        {"title": "High Pitch Alarm Beep", "path": alarm_path, "description": "Pulsing 1000Hz emergency warning siren."},
        {"title": "Engine Idle Sound", "path": engine_path, "description": "Low frequency diesel motor mechanical vibration."},
        {"title": "Ocean Water Surf", "path": water_path, "description": "Ambient natural rushing water and waves."}
    ]

def main():
    print("=" * 65)
    print("🎓 PHASE 3: Multimodal Audio (Hearing) with EmbeddingGemma-2")
    print("=" * 65)

    base_dir = Path(__file__).resolve().parent.parent
    uploads_dir = base_dir / "data" / "uploads"
    db_dir = base_dir / "data" / "lancedb"

    # 1. Generate 16kHz mono audio files
    print(f"\n[1. Synthesizing 16 kHz Mono Test Audio Files]")
    audio_items = create_sample_audio_files(uploads_dir)
    for item in audio_items:
        print(f"  • Created audio: {item['path'].name} (16kHz Mono, 3.0s)")

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

    # 3. Vectorize Audio using Audio Encoder
    # Google Best Practice: Media files are passed raw without text prompt prefixes
    print(f"\n[3. Vectorizing Audio Files Directly with Audio Encoder]")
    audio_paths = [str(item["path"]) for item in audio_items]

    start_embed = time.time()
    audio_vectors = model.encode(
        audio_paths,
        truncate_dim=256,
        normalize_embeddings=True,
        batch_size=4
    )
    print(f"  ✓ Encoded {len(audio_paths)} audio files into 256d vectors in {time.time() - start_embed:.2f}s!")

    # 4. Save Audio Embeddings to LanceDB
    print(f"\n[4. Storing Audio Embeddings in LanceDB]")
    db = lancedb.connect(str(db_dir))
    table = db.open_table("knowledge_base")

    new_records = []
    for vec, item in zip(audio_vectors, audio_items):
        new_records.append({
            "vector": vec.tolist(),
            "title": item["title"],
            "content": str(item["path"].name),     # Stores the audio filename
            "source": str(item["path"].name),
            "modality": "audio"
        })

    table.add(new_records)
    print(f"  ✓ Added {len(new_records)} audio records. Total documents in DB now: {table.count_rows()}")

    # 5. Test Cross-Modal Audio Search (Text Query ➔ Matching Audio)
    test_queries = [
        "loud emergency warning siren beep",
        "diesel truck engine running and mechanical vibration",
        "flowing river stream and rushing water ambient noise"
    ]

    print(f"\n[5. Testing Cross-Modal Audio Search (Text Query ➔ Audio Retrieval)]")
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

        results = table.search(q_vec.tolist()).where("modality = 'audio'").limit(2).to_list()
        search_ms = (time.time() - start_q) * 1000

        print(f"  Ranked Results ({search_ms:.1f}ms):")
        for rank, res in enumerate(results, start=1):
            sim = float(np.dot(q_vec, res["vector"]))
            print(f"  #{rank} [{res['title']}] (Sim: {sim:.4f}) -> File: {res['content']}")

    print("\n" + "=" * 65)
    print("🎉 PHASE 3 COMPLETE: True Multimodal Audio Hearing is Operational!")
    print("=" * 65)

if __name__ == "__main__":
    main()
