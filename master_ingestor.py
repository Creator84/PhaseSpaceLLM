"""Master ingestion script.

Supports incremental ingestion into a shared brain (global_swarm_brain.pth) with a
manifest file that tracks what has already been learned.

You can run this script repeatedly, adding new sources over time, and the brain
will steadily accumulate knowledge without re-processing the same data.

Examples:
  python3 master_ingestor.py --pdf book1.pdf
  python3 master_ingestor.py --pdf book2.pdf
  python3 master_ingestor.py --zim bulbagarden_en_all_nopic.zim --keywords science,tech
"""

import argparse
import json
import os
from datetime import datetime, timezone

from zim_ingestor import ZimSwarmIngestor
from src.os2_wrapper import PhaseSpaceMemoryNode

BRAIN_FILE = "global_swarm_brain.pth"
MANIFEST_FILE = "ingestion_manifest.json"


def load_manifest(path=MANIFEST_FILE):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest, path=MANIFEST_FILE):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def file_signature(path):
    stat = os.stat(path)
    return f"{stat.st_mtime_ns}-{stat.st_size}"


def should_ingest_file(manifest, category, path):
    entry = manifest.get(category, {}).get(path)
    if not entry:
        return True
    return entry.get("sig") != file_signature(path)


def mark_file_ingested(manifest, category, path):
    manifest.setdefault(category, {})[path] = {
        "sig": file_signature(path),
        # Use timezone-aware UTC timestamps (avoids Python deprecation warning)
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def ingest_pdf(brain, manifest, pdf_path, batch_size=128):
    """Extract text from a PDF and learn it into the brain in chunks."""
    from PyPDF2 import PdfReader

    if not os.path.exists(pdf_path):
        print(f"[!] PDF not found: {pdf_path}")
        return

    if not should_ingest_file(manifest, "pdf", pdf_path):
        print(f"[✓] Already ingested PDF: {pdf_path} (skipping)")
        return

    print(f"[*] Ingesting PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    pages = [p.extract_text() or "" for p in reader.pages]
    doc_text = "\n".join(pages)

    # Simple word-based chunking with overlap
    words = doc_text.split()
    chunk_size = 1200
    overlap = 200

    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            stamped_chunk = f"[SOURCE: {os.path.basename(pdf_path)} | FORMAT: PDF Chunk]\n{chunk}"
            chunks.append(stamped_chunk)
        i += chunk_size - overlap

    print(f"    -> {len(chunks)} chunks generated")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        brain.batch_learn(batch)
        print(
            f"    -> Learned PDF chunk batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}"
        )

    mark_file_ingested(manifest, "pdf", pdf_path)


def ingest_txt(brain, manifest, txt_path, batch_size=128):
    if not os.path.exists(txt_path):
        print(f"[!] Text file not found: {txt_path}")
        return

    if not should_ingest_file(manifest, "txt", txt_path):
        print(f"[✓] Already ingested text: {txt_path} (skipping)")
        return

    print(f"[*] Ingesting text file: {txt_path}")
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Simple chunking
    words = text.split()
    chunk_size = 1200
    overlap = 200

    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            stamped_chunk = f"[SOURCE: {os.path.basename(txt_path)} | FORMAT: Text Chunk]\n{chunk}"
            chunks.append(stamped_chunk)
        i += chunk_size - overlap

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        brain.batch_learn(batch)
        print(
            f"    -> Learned text chunk batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}"
        )

    mark_file_ingested(manifest, "txt", txt_path)


def ingest_zim(
    brain,
    manifest,
    zim_path,
    keywords=None,
    limit=None,
    batch_size=128,
    num_workers=None,
):
    if not os.path.exists(zim_path):
        print(f"[!] ZIM not found: {zim_path}")
        return

    manifest.setdefault("zim", {})
    zim_manifest = manifest["zim"].get(zim_path, {})
    seen_ids = set(zim_manifest.get("ingested_entry_ids", []))

    ingestor = ZimSwarmIngestor(zim_path)
    ingestor.memory = brain

    def _progress_callback(processed, total):
        # Persist progress for safe restarts / crash recovery
        zim_manifest["last_processed"] = processed
        zim_manifest["total_entries"] = total
        zim_manifest["ingested_entry_ids"] = list(seen_ids)
        manifest["zim"][zim_path] = zim_manifest
        save_manifest(manifest)

    print(f"[*] Ingesting ZIM: {zim_path} (already learned {len(seen_ids)} entries)")
    ingestor.ingest(
        limit=limit,
        keywords=keywords,
        batch_size=batch_size,
        num_workers=num_workers,
        seen_entry_ids=seen_ids,
        progress_callback=_progress_callback,
    )

    # Persist final state
    zim_manifest["ingested_entry_ids"] = list(seen_ids)
    zim_manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    manifest["zim"][zim_path] = zim_manifest


def main():
    parser = argparse.ArgumentParser(description="Ingest data sources into the PhaseSpace brain.")
    parser.add_argument("--zim", nargs="*", help="ZIM files to ingest")
    parser.add_argument("--pdf", nargs="*", help="PDF files to ingest")
    parser.add_argument("--txt", nargs="*", help="Text files to ingest")
    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        help="Comma-separated keywords to filter ZIM ingestion (e.g., 'science,ai,tech')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of new items to learn from a ZIM source",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for learning (smaller uses less RAM, larger is faster)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for ZIM ingestion",
    )
    args = parser.parse_args()

    keywords = [k.strip() for k in (args.keywords or "").split(",") if k.strip()]
    if not keywords:
        keywords = None

    print("--- Loading brain & manifest ---")
    brain = PhaseSpaceMemoryNode()
    brain.load_brain(BRAIN_FILE)
    manifest = load_manifest()

    try:
        if args.pdf:
            for pdf in args.pdf:
                ingest_pdf(brain, manifest, pdf, batch_size=args.batch_size)

        if args.txt:
            for txt in args.txt:
                ingest_txt(brain, manifest, txt, batch_size=args.batch_size)

        if args.zim:
            for zim in args.zim:
                ingest_zim(
                    brain,
                    manifest,
                    zim,
                    keywords=keywords,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )

    finally:
        print("--- Persisting brain & manifest ---")
        brain.save_brain(BRAIN_FILE)
        save_manifest(manifest)

    print("--- Ingestion complete ---")


if __name__ == "__main__":
    main()