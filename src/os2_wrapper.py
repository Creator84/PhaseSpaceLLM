import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from src.orbit_core import PhaseSpaceOrbit

class PhaseSpaceMemoryNode:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', alpha=0.8):
        print(f"[OS²] Booting Local Memory Node (Alpha={alpha})...")
        self.embedder = SentenceTransformer(embedding_model)
        self.orbit = PhaseSpaceOrbit(state_dim=384, alpha=alpha)
        self.knowledge_vault = [] # Stores (raw_text, normalized_tensor)

    def learn(self, text_content: str):
        vector = torch.tensor(self.embedder.encode(text_content)).unsqueeze(0)
        normalized_vector = F.normalize(vector, p=2, dim=1)
        
        # Save to vault for exact retrieval later
        self.knowledge_vault.append((text_content, normalized_vector))
        
        # Evolve the Swarm's global state
        self.orbit.update_orbit(normalized_vector)
        return len(self.knowledge_vault)

    def batch_learn(self, text_list: list):
        print(f"[OS²] Batch learning {len(text_list)} data points...")
        for text in text_list:
            self.learn(text)
        print(f"      -> Swarm memory now contains {len(self.knowledge_vault)} facts.")

    def recall(self, objective: str, top_k: int = 2):
        if not self.knowledge_vault:
            return ["Memory is empty."]

        # 1. Steer the Orbit
        objective_vector = torch.tensor(self.embedder.encode(objective)).unsqueeze(0)
        steered_state = self.orbit.update_orbit(objective_vector)
        clean_objective = F.normalize(objective_vector, p=2, dim=1)

        # 2. Filter the Vault
        scored_facts = []
        for text, norm_vector in self.knowledge_vault:
            similarity = F.cosine_similarity(clean_objective, norm_vector).item()
            scored_facts.append((similarity, text))

        # 3. Return the best matches
        scored_facts.sort(reverse=True, key=lambda x: x[0])
        return [fact for score, fact in scored_facts[:top_k]]
        
    def wipe_memory(self):
        self.orbit.h_t = torch.zeros(1, 384)
        self.knowledge_vault = []
        print("[OS²] Memory wiped. Ready for new mission.")
