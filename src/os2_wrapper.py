import os
import pickle
import sqlite3
import numpy as np
import torch
import faiss
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.orbit_core import PhaseSpaceOrbit

class PhaseSpaceMemoryNode:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2', alpha=0.8, db_path="swarm_vault.db"):
        print(f"[OS²] Booting Enterprise Memory Node (Alpha={alpha})...")
        
        self.db_path = db_path
        self._init_sqlite()

        # Base Embedding (The "Fast Brain")
        self.embedder = SentenceTransformer(embedding_model)
        self.orbit = PhaseSpaceOrbit(state_dim=384, alpha=alpha)

        # Cross-Encoder (The "Slow, Careful Brain")
        print(f"[OS²] Booting Cross-Encoder Reranker ({reranker_model})...")
        self.reranker = CrossEncoder(reranker_model)

        # HNSW FAISS Index (Graph-Based Search)
        self._build_or_load_faiss()

    def _init_sqlite(self):
        """Creates a disk-backed database for raw text so it doesn't crash your RAM."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS vault (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def _build_or_load_faiss(self):
        """Initializes the HNSW Graph Index."""
        self.index_file = "swarm_faiss.index"
        if os.path.exists(self.index_file):
            print("[OS²] Loading existing HNSW FAISS index...")
            self.faiss_index = faiss.read_index(self.index_file)
        else:
            print("[OS²] Creating new HNSW FAISS index...")
            # M=32 is the number of connections per node in the graph. Highly accurate.
            base_index = faiss.IndexHNSWFlat(self.orbit.state_dim, 32)
            # We must wrap it in an IDMap so we can link FAISS IDs to SQLite IDs
            self.faiss_index = faiss.IndexIDMap2(base_index)

    def learn(self, text_content: str):
        self.batch_learn([text_content])

    def batch_learn(self, text_list: list):
        if not text_list:
            return

        # 1. Calculate next available IDs for SQLite
        self.cursor.execute("SELECT MAX(id) FROM vault")
        max_id = self.cursor.fetchone()[0]
        start_id = (max_id or 0) + 1
        
        # 2. Prepare Data
        ids = np.arange(start_id, start_id + len(text_list), dtype=np.int64)
        db_data = [(int(idx), text) for idx, text in zip(ids, text_list)]
        
        # 3. Save Text to SSD (SQLite)
        self.cursor.executemany("INSERT INTO vault (id, text) VALUES (?, ?)", db_data)
        self.conn.commit()

        # 4. Save Math to RAM (FAISS)
        vectors = [torch.tensor(self.embedder.encode(t)).unsqueeze(0) for t in text_list]
        normalized = [F.normalize(v, p=2, dim=1) for v in vectors]
        vectors_np = np.vstack([v.numpy().astype('float32') for v in normalized])
        
        self.faiss_index.add_with_ids(vectors_np, ids)
        
        print(f"\r    -> Learned batch of {len(text_list)}. Total items in Swarm: {self.faiss_index.ntotal}", end="")

    def recall(self, objective: str, top_k: int = 2):
        if self.faiss_index.ntotal == 0:
            return ["Memory is empty."]

        # 1. Steer the Orbit
        objective_vector = torch.tensor(self.embedder.encode(objective)).unsqueeze(0)
        self.orbit.update_orbit(objective_vector)
        clean_objective = F.normalize(objective_vector, p=2, dim=1).numpy().astype('float32')

        # Stage 1: Fast Graph Retrieval (Pull 10x the requested amount)
        fetch_k = min(top_k * 10, self.faiss_index.ntotal)
        
        # FAISS searches the graph in milliseconds
        distances, indices = self.faiss_index.search(clean_objective, fetch_k)
        
        # Fetch only the winning texts directly from the SSD
        valid_ids = [int(idx) for idx in indices[0] if idx != -1]
        if not valid_ids:
            return ["No relevant memories found."]

        placeholders = ','.join('?' for _ in valid_ids)
        self.cursor.execute(f"SELECT text FROM vault WHERE id IN ({placeholders})", valid_ids)
        
        # We have to re-sort the SQL results to match the FAISS distance order
        fetched_rows = self.cursor.fetchall()
        raw_results = [row[0] for row in fetched_rows]

        # Stage 2: Cross-Encoder Reranking
        pairs = [[objective, doc] for doc in raw_results]
        scores = self.reranker.predict(pairs)
        scored_results = sorted(zip(scores, raw_results), key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_results[:top_k]]

    def save_brain(self, filename="swarm_orbit.pth"):
        """Saves the Orbit state and writes the FAISS index to disk."""
        # Save Orbit
        data = {'orbit_state': self.orbit.h_t}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            
        # Save FAISS Graph
        faiss.write_index(self.faiss_index, self.index_file)
        print(f"\n[OS²] Brain saved! Orbit in {filename}, Graph in {self.index_file}, Text in {self.db_path}.")

    def load_brain(self, filename="swarm_orbit.pth"):
        """Restores the Swarm Orbit (SQLite and FAISS load automatically in __init__)."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.orbit.h_t = data['orbit_state']
            print(f"[OS²] Orbit state loaded. Swarm ready with {self.faiss_index.ntotal} facts.")
            return True
        else:
            print("[!] No prior orbit found. Starting fresh.")
            return False

    def wipe_memory(self):
        self.orbit.h_t = torch.zeros(1, 384)
        self.cursor.execute("DROP TABLE IF EXISTS vault")
        self.conn.commit()
        self._init_sqlite()
        
        base_index = faiss.IndexHNSWFlat(self.orbit.state_dim, 32)
        self.faiss_index = faiss.IndexIDMap2(base_index)
        
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
            
        print("[OS²] Memory wiped entirely (SQLite & FAISS reset).")