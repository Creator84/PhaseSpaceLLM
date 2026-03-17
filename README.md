# PhaseSpace LLM – Persistent Semantic Memory for Local LLMs

A lightweight, local-first memory system that maintains a **continuous "orbit"** (EMA-updated point) in semantic embedding space while storing exact text chunks for retrieval. Supports **incremental ingestion** from multiple sources (PDF, TXT, ZIM, APIs) with a persistent manifest to avoid re-processing.

## Core Idea

Instead of a full vector database:
- New facts/documents are embedded (all-MiniLM-L6-v2) and smoothly incorporated into a single persistent latent state via exponential moving average (EMA).
- This "orbit" acts as a compressed semantic attractor / summary of everything seen.
- For synthesis or Q&A: cosine-rank chunks against the current objective → feed top resonant facts to a local LLM (Llama 3.2 via Ollama).

→ Extremely low overhead, no indexing latency, runs forever in RAM or saved to disk.

## Features

- **Persistent orbit state** (save/load `.pth` files)
- **Incremental ingestion** with manifest tracking (no duplicate processing)
- **Multi-source support**: PDF, TXT, ZIM archives, extensible to APIs/databases
- Optional FAISS-based nearest-neighbor recall for sub-millisecond retrieval
- **Parallel ingest** — Uses all available CPU cores for 10-20x faster processing
- API server + browser UI for injecting facts & synthesizing reports
- Batch document ingestion (PDF/TXT) with smart chunking and overlap
- ZIM archive ingestion support (Wikipedia / Kiwix)
- Strict RAG-style answers ("document does not specify" when no match)

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/PhaseSpaceLLM.git
cd PhaseSpaceLLM
```

### 2. Set Up Virtual Environment
```bash
# Create and activate virtual environment
python3 -m venv swarm_env
source swarm_env/bin/activate  # On Windows: swarm_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and Start Ollama (for LLM queries)
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required model
ollama pull llama3.2

# Start Ollama server (in background)
ollama serve
```

## Quick Start Tutorial

### Step 1: Start the API Server
```bash
uvicorn src.api_server:app --reload --port 8000
```
→ Open http://localhost:8000 in your browser

### Step 2: Ingest Your First Document
```bash
# Ingest a PDF book
python3 master_ingestor.py --pdf mybook.pdf

# Ingest a text file
python3 master_ingestor.py --txt mydocument.txt
```

### Step 3: Query the Knowledge Base
- Use the web UI at http://localhost:8000
- Or query programmatically via the API

### Step 4: Add More Knowledge Incrementally
```bash
# Add another book (manifest prevents re-processing previous files)
python3 master_ingestor.py --pdf anotherbook.pdf

# Add ZIM content with keyword filtering
python3 master_ingestor.py --zim wikipedia.zim --keywords science,technology
```

## Advanced Usage

### CLI Options
```bash
python3 master_ingestor.py --help
```

**Common commands:**
```bash
# Ingest multiple PDFs at once
python3 master_ingestor.py --pdf book1.pdf book2.pdf

# Ingest ZIM with limits and keywords
python3 master_ingestor.py --zim wikipedia.zim --keywords ai,ml --limit 1000

# Control batch size for memory management
python3 master_ingestor.py --pdf largebook.pdf --batch-size 64

# Use more CPU cores for faster processing
python3 master_ingestor.py --zim archive.zim --num-workers 16
```

### Manifest System
- **What it does**: Tracks which files have been ingested and when
- **File**: `ingestion_manifest.json`
- **Benefits**: 
  - Skip already-processed files
  - Safe restart after crashes
  - Incremental knowledge accumulation

### Extending to New Sources
The ingestion system is designed to be extensible. To add a new source type:

1. Create a new `ingest_*()` function in `master_ingestor.py`
2. Use the same pattern: check manifest, process in chunks, call `brain.batch_learn()`
3. Add CLI argument parsing in `main()`

Example for API ingestion:
```python
def ingest_api(brain, manifest, api_url, batch_size=128):
    # Fetch data from API
    # Chunk and learn
    # Update manifest
```

### Interactive Dashboard
```bash
python dashboard.py
```
→ Open http://localhost:8001 for 3D orbit visualization and queries

## Project Structure
- `src/orbit_core.py` — EMA-based phase-space orbit logic
- `src/api_server.py` — FastAPI + UI for interaction
- `src/os2_wrapper.py` — Persistent memory node (with optional FAISS recall)
- `master_ingestor.py` — **NEW**: Incremental multi-source ingestion CLI with manifest
- `ingestion_manifest.json` — **NEW**: Tracks ingested files and progress
- `zim_ingestor.py` — ZIM archive ingestion logic (Wikipedia/Kiwix)
- `ingest.py` — Legacy single-document ingestion
- `dashboard.py` — 3D orbit visualization + query UI
- `global_swarm_brain.pth` — Persistent brain file (created after first ingestion)
- `examples/` — Sample scripts and test files

## Troubleshooting

### WSL Crashes During Large ZIM Ingestion
- **Cause**: Memory overflow from loading all entries at once
- **Fix**: The new streaming ingestion prevents this. If still crashing, reduce `--batch-size`

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Slow Ingestion
- Increase `--num-workers` to use more CPU cores
- Reduce `--batch-size` for lower memory usage (but slower)
- For very large files, process in multiple sessions

### Manifest Errors
- Delete `ingestion_manifest.json` to force re-ingestion (loses progress tracking)
- Check file permissions on the manifest file

## Performance Tips

- **Memory**: Smaller batch sizes use less RAM but are slower
- **Speed**: More workers = faster processing (up to your CPU core count)
- **Storage**: Brain files grow linearly with content; compress old manifests if needed
- **Query speed**: FAISS index provides ~100x faster retrieval for large knowledge bases

## Roadmap / Ideas
- Improve ingestion speed / parallelization (especially for large ZIMs)
- Add configurable decay/forgetting to age out old facts
- Add topic-specific orbit instances and switchable contexts
- Add optional persistent FAISS index file for faster startup
- Support for more document formats (DOCX, HTML, etc.)
- Web crawler integration for automatic knowledge gathering

MIT licensed – feel free to fork & experiment!