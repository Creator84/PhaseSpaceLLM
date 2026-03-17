import re
import time
import os
from concurrent.futures import ProcessPoolExecutor
from libzim.reader import Archive
from src.os2_wrapper import PhaseSpaceMemoryNode
from bs4 import BeautifulSoup

# Global cleanup function for parallel processing
def _clean_html_worker(html_text):
    """Worker function to clean HTML in parallel."""
    soup = BeautifulSoup(html_text, "html.parser")
    for script_or_style in soup(["script", "style", "meta", "link"]):
        script_or_style.decompose()
    for junk in soup.find_all(class_=["infobox", "wb-edithandle", "mw-jump-link", "navbox"]):
        junk.decompose()
    text = soup.get_text(separator=' ')
    clean_text = re.sub(r'\s+', ' ', text).strip()
    return clean_text if len(clean_text) >= 500 else ""

def _fetch_entry_worker(args):
    """Worker to fetch entry from ZIM by index (each worker opens its own archive)."""
    zim_path, entry_id = args
    try:
        archive = Archive(zim_path)
        entry = archive._get_entry_by_id(entry_id)
        item = entry.get_item()
        if item.mimetype == "text/html":
            raw_html = bytes(item.content).decode('utf-8', errors='ignore')
            title = entry.title if hasattr(entry, 'title') else f"#{entry_id}"
            return (entry_id, raw_html, title)
    except Exception:
        pass
    return None


class ZimSwarmIngestor:
    def __init__(self, zim_path):
        if not os.path.exists(zim_path):
            raise FileNotFoundError(f"ZIM file not found at {zim_path}")
        
        print(f"[OS²] Opening ZIM Archive: {zim_path}")
        self.archive = Archive(zim_path)
        self.memory = PhaseSpaceMemoryNode()

    def clean_html(self, html_text):
        """Deep cleans Wikipedia/ZIM HTML to extract only meaningful text."""
        soup = BeautifulSoup(html_text, "html.parser")

        # 1. Kill CSS, Scripts, and Meta-Data tags
        for script_or_style in soup(["script", "style", "meta", "link"]):
            script_or_style.decompose()

        # 2. Kill the Wikipedia-specific "Infobox" and "Edit" handles you saw
        for junk in soup.find_all(class_=["infobox", "wb-edithandle", "mw-jump-link", "navbox"]):
            junk.decompose()

        # 3. Get text and clean whitespace
        text = soup.get_text(separator=' ')
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        # 4. Filter for length: Articles under 500 chars are usually just lists or stubs
        if len(clean_text) < 500:
            return ""
            
        return clean_text


    def ingest(
        self,
        limit=None,
        keywords=None,
        batch_size=128,
        num_workers=None,
        seen_entry_ids=None,
        progress_callback=None,
    ):
        """Ingest ZIM articles with parallel HTML cleaning and streaming to prevent memory overflow.

        Args:
            limit: Max articles to ingest (None = all)
            keywords: Filter by keywords (None = all)
            batch_size: Articles to batch-learn at once
            num_workers: CPU cores to use (default: auto-detect, up to 24)
            seen_entry_ids: Optional set of entry IDs already ingested (increments work)
            progress_callback: Optional callable(processed_count, total_entries) called periodically
        """
        import time

        if keywords is None:
            keywords = []
        if num_workers is None:
            num_workers = min(os.cpu_count() or 8, 24)  # Cap at 24 cores
        if seen_entry_ids is None:
            seen_entry_ids = set()

        print(f"[*] Ingestion mode: STREAMING PARALLEL ({num_workers} cores) | Batch size: {batch_size}")
        print(f"[*] Targeting: {keywords or 'ALL'}")

        processed_count = 0
        total_entries = self.archive.all_entry_count
        start_time = time.time()

        article_bucket = []  # Batch for learning
        chunk_size = batch_size * 4  # Process in larger chunks to amortize I/O

        print(f"\n[STREAMING] Processing {total_entries} entries in chunks...")

        for chunk_start in range(0, total_entries, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_entries)

            # Create fetch tasks for this chunk only
            fetch_tasks = [(self.archive.filename, i) for i in range(chunk_start, chunk_end)]

            # Fetch and clean in parallel within this chunk
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                entry_data = []

                # Fetch entries in this chunk
                for result in executor.map(_fetch_entry_worker, fetch_tasks):
                    if result:
                        entry_id, raw_html, title = result
                        # Skip already seen entries
                        if entry_id in seen_entry_ids:
                            continue
                        entry_data.append((entry_id, title, raw_html))

                # Clean entries in this chunk in parallel
                futures = [executor.submit(_clean_html_worker, raw) for _, _, raw in entry_data]

                for (entry_id, title, _), future in zip(entry_data, futures):
                    cleaned = future.result()
                    if not cleaned:
                        continue

                    # Apply keyword filtering
                    if keywords and not any(word.lower() in cleaned.lower() for word in keywords):
                        continue

                    seen_entry_ids.add(entry_id)
                    if keywords and not any(word.lower() in cleaned.lower() for word in keywords):
                        continue
                    
                    # --- ADD THIS SOURCE TRACKER ---
                    # We inject the filename and article title directly into the text
                    stamped_text = f"[SOURCE: {os.path.basename(self.archive.filename)} | TITLE: {title}]\n{cleaned}"
                    
                    seen_entry_ids.add(entry_id)
                    article_bucket.append(stamped_text) # Learn the stamped version!
                    processed_count += 1

                    # Batch learn
                    if len(article_bucket) >= batch_size:
                        self.memory.batch_learn(article_bucket)
                        elapsed = time.time() - start_time
                        speed = processed_count / elapsed if elapsed > 0 else 0
                        print(f"    [{processed_count}/{total_entries}] Learned batch | ~{speed:.1f} articles/sec")
                        article_bucket = []

                    if progress_callback:
                        progress_callback(processed_count, total_entries)

                    if limit is not None and processed_count >= limit:
                        break

                if limit is not None and processed_count >= limit:
                    break

                elapsed = time.time() - start_time
                speed = (chunk_end) / elapsed if elapsed > 0 else 0
                print(f"    [{chunk_end}/{total_entries}] Chunk complete | ~{speed:.1f} entries/sec")

        # Final flush
        if article_bucket:
            self.memory.batch_learn(article_bucket)

        elapsed = time.time() - start_time
        final_speed = processed_count / elapsed if elapsed > 0 else 0
        print(f"\n[✓] DONE. Ingested {processed_count} articles in {elapsed:.2f}s (~{final_speed:.1f} articles/sec).")
        return self.memory

# ==========================================
# UNIT TEST LOGIC (Only runs if you execute this file directly)
# ==========================================
if __name__ == "__main__":
    # Use a small limit for a quick sanity check
    TEST_ZIM = "wikipedia_nb_all_mini_2026-03.zim" 
    
    if os.path.exists(TEST_ZIM):
        ingestor = ZimSwarmIngestor(TEST_ZIM)
        # We use a tiny limit of 5 just to make sure the math and cleaning works
        test_brain = ingestor.ingest(limit=5, keywords=["norge", "tech"]) 
        print("[✓] Unit test complete. Class is functional.")
    else:
        print(f"[*] ZimSwarmIngestor class loaded. (Test file {TEST_ZIM} not found).")
