import sys
import time
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from app.core import MultimodalRAGEngine
from app.config import settings
from app.processors import PROCESSOR_REGISTRY

def main():
    print("=" * 65)
    print("📂 Multimodal Folder Ingestion Tool")
    print("=" * 65)

    folder_path = settings.UPLOADS_DIR
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"\nScanning folder: {folder_path.resolve()}")

    engine = MultimodalRAGEngine()
    
    # Collect all supported files
    supported_extensions = set(PROCESSOR_REGISTRY.keys())
    files = [
        f for f in folder_path.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions and not f.name.endswith("_16k_mono.wav")
    ]

    if not files:
        print(f"\n⚠️  No supported files found in {folder_path}!")
        print(f"Supported formats: {', '.join(sorted(supported_extensions))}")
        print("\nDrop your files (.txt, .pdf, .md, .png, .jpg, .wav, .mp3, .mp4) in that folder and run this script again!")
        return

    print(f"Found {len(files)} files to index:")
    for f in files:
        print(f"  • {f.name} ({f.suffix.upper()})")

    print("\n[Indexing files into LanceDB Knowledge Base...]")
    start_total = time.time()
    total_chunks = 0

    for idx, f in enumerate(files, start=1):
        try:
            start_f = time.time()
            chunks = engine.ingest_file(f)
            total_chunks += chunks
            print(f"  [{idx}/{len(files)}] ✓ {f.name} -> {chunks} vectors ({time.time() - start_f:.2f}s)")
        except Exception as e:
            print(f"  [{idx}/{len(files)}] ✗ Failed to index {f.name}: {e}")

    total_in_db = engine.vector_store.count()
    print("\n" + "=" * 65)
    print(f"🎉 INGESTION COMPLETE in {time.time() - start_total:.2f}s!")
    print(f"• Ingested {total_chunks} new vectors.")
    print(f"• Total documents in LanceDB now: {total_in_db}")
    print("=" * 65)

if __name__ == "__main__":
    main()
