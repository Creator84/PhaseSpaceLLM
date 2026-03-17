import torch
import torch.nn.functional as F
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from src.orbit_core import PhaseSpaceOrbit

app = FastAPI(title="OS² Phase-Space Swarm API")

print("Loading Global Swarm Brain...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
swarm_orbit = PhaseSpaceOrbit(state_dim=384)

knowledge_vectors = []

class InjectRequest(BaseModel):
    text: str

class SynthesizeRequest(BaseModel):
    objective: str = "technical and pricing strategy"

@app.post("/inject")
async def inject_memory(req: InjectRequest):
    """Takes text, converts to math, and updates the Swarm's Orbit."""
    if not req.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    vector = torch.tensor(embedder.encode(req.text)).unsqueeze(0)
    knowledge_vectors.append((req.text, vector))
    
    swarm_orbit.update_orbit(vector)
    
    return {"status": "success", "message": f"Fact injected! Swarm memory size: {len(knowledge_vectors)} facts."}

@app.post("/synthesize")
async def synthesize_report(req: SynthesizeRequest):
    if not knowledge_vectors:
        return {"report": "The Swarm's memory is empty. Inject data first."}

    print(f"\n[+] Steering Swarm toward objective: {req.objective}")
    
    objective_vector = torch.tensor(embedder.encode(req.objective)).unsqueeze(0)
    steered_state = swarm_orbit.update_orbit(objective_vector)

    scored_facts = []
    
    clean_objective = F.normalize(objective_vector, p=2, dim=1)

    for text, vector in knowledge_vectors:
        normalized_vector = F.normalize(vector, p=2, dim=1)
        
        similarity = F.cosine_similarity(clean_objective, normalized_vector).item()
        scored_facts.append((similarity, text))

    scored_facts.sort(reverse=True, key=lambda x: x[0])
    resonating_facts = [fact for score, fact in scored_facts[:2]]
    
    context_string = "\n".join(resonating_facts)

    system_prompt = f"""
    You are the Synthesizer Node of an autonomous AI swarm. 
    The Swarm's mathematical orbit has isolated the following resonant data points:

    {context_string}

    Write a 3-sentence executive summary focused on: {req.objective}.
    Base it ONLY on this data. 
    """

    try:
        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'user', 'content': system_prompt}
        ])
        report = response['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "resonant_facts_used": resonating_facts,
        "report": report
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves a sleek HTML dashboard to interact with the Swarm API."""
    html_content = """
    <html>
        <head><title>OS² Swarm Interface</title></head>
        <body style="font-family: Arial, sans-serif; padding: 20px; max-width: 600px; margin: auto; background-color: #f9f9f9;">
            <h2 style="color: #333;">🧠 Phase-Space Swarm Control</h2>
            
            <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h3>1. Inject Memory</h3>
                <input type="text" id="memoryText" style="width: 75%; padding: 8px;" placeholder="e.g., The CEO loves pizza...">
                <button onclick="inject()" style="padding: 8px 15px; background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">Inject</button>
                <p id="injectStatus" style="color: green; font-size: 0.9em; margin-top: 10px;"></p>

                <hr style="margin: 20px 0; border: 0; border-top: 1px solid #eee;">

                <h3>2. Synthesize Swarm State</h3>
                <input type="text" id="objectiveText" style="width: 75%; padding: 8px;" value="technical and pricing strategy">
                <button onclick="synthesize()" style="padding: 8px 15px; background: #28A745; color: white; border: none; border-radius: 4px; cursor: pointer;">Synthesize</button>
                
                <div id="reportBox" style="margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px; white-space: pre-wrap; min-height: 100px; font-size: 0.95em; color: #333;">
                    <i>Awaiting command...</i>
                </div>
            </div>

            <script>
                async function inject() {
                    const text = document.getElementById('memoryText').value;
                    const res = await fetch('/inject', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text})
                    });
                    const data = await res.json();
                    document.getElementById('injectStatus').innerText = data.message;
                    document.getElementById('memoryText').value = '';
                }

                async function synthesize() {
                    document.getElementById('reportBox').innerHTML = "<i>Swarm is thinking (Llama 3.2 offline)...</i>";
                    const obj = document.getElementById('objectiveText').value;
                    const res = await fetch('/synthesize', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({objective: obj})
                    });
                    const data = await res.json();
                    let output = "<b>🎯 RESONANT FACTS USED:</b>\\n" + data.resonant_facts_used.join("\\n") + "\\n\\n";
                    output += "<b>📝 FINAL REPORT:</b>\\n" + data.report;
                    document.getElementById('reportBox').innerHTML = output;
                }
            </script>
        </body>
    </html>
    """
    return html_content
