import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.decomposition import PCA
from src.os2_wrapper import PhaseSpaceMemoryNode
from pydantic import BaseModel
import ollama
import logging
import torch
import torch.nn.functional as F

# Suppress FastAPI access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your Enterprise Brain
memory = PhaseSpaceMemoryNode()
memory.load_brain("global_swarm_brain.pth")

pca = PCA(n_components=3)
LATEST_QUERY_VECTOR = None

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return """
    <html>
        <head>
            <title>OS² Enterprise Console</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"></script>
            <style>
                body { margin: 0; background: #05070a; color: #00ff41; font-family: 'Courier New', monospace; overflow: hidden; }
                #chart { width: 100vw; height: 100vh; position: absolute; z-index: 0; }
                
                .hud-top-left { position: absolute; top: 15px; left: 15px; z-index: 10; font-size: 1.2em; font-weight: bold; text-shadow: 0 0 5px #00ff41; }
                .hud-top-right { position: absolute; top: 15px; right: 15px; z-index: 10; text-align: right; font-size: 0.9em; background: rgba(0,255,65,0.1); padding: 10px; border: 1px solid #00ff41; border-radius: 4px;}
                
                /* Upgraded UI Container */
                .ui-wrapper { position: absolute; bottom: 0; width: 100%; z-index: 10; display: flex; flex-direction: column; }
                .ui-minimize-bar { background: rgba(5, 7, 10, 0.9); border-top: 1px solid #00ff41; text-align: center; padding: 5px; cursor: pointer; font-size: 0.8em; font-weight: bold; transition: background 0.2s; }
                .ui-minimize-bar:hover { background: #002200; color: #fff; }
                
                .ui-bottom { background: rgba(5, 7, 10, 0.9); padding: 20px; box-sizing: border-box; display: flex; gap: 10px; align-items: flex-start; transition: all 0.3s ease; }
                .ui-hidden { display: none !important; }
                
                .search-container { flex-grow: 1; }
                input[type="text"] { background: #000; color: #00ff41; border: 1px solid #00ff41; padding: 12px; width: 100%; font-family: monospace; font-size: 1.1em; box-sizing: border-box; outline: none; }
                input[type="text"]:focus { box-shadow: 0 0 10px #00ff41; }
                
                button { background: #002200; color: #00ff41; border: 1px solid #00ff41; padding: 12px 20px; cursor: pointer; font-family: monospace; font-weight: bold; transition: all 0.2s; margin-bottom: 5px; width: 100%;}
                button:hover { background: #00ff41; color: #000; }
                .btn-danger { background: #330000; color: #ff003c; border-color: #ff003c; }
                .btn-danger:hover { background: #ff003c; color: #fff; box-shadow: 0 0 15px #ff003c; }

                #output { margin-top: 15px; font-size: 1em; color: #fff; line-height: 1.4; white-space: pre-wrap; }
                #xray-panel { display: none; margin-top: 15px; padding: 10px; background: rgba(255, 255, 255, 0.05); border: 1px dashed #aaa; color: #aaa; font-size: 0.85em; max-height: 200px; overflow-y: auto; white-space: pre-wrap; }
                .source-tag { color: #ffaa00; font-weight: bold; }
            </style>
        </head>
        <body>
            <div id="chart"></div>
            
            <div class="hud-top-left">OS² // PHASE-SPACE SWARM</div>
            <div class="hud-top-right" id="telemetry">
                <div>SYS.STATUS: <span id="status-text">ONLINE</span></div>
                <div>NODES: <span id="node-count">0</span></div>
                <div>ORBIT: <span id="orbit-coords">X:0.0 Y:0.0 Z:0.0</span></div>
            </div>

            <div class="ui-wrapper">
                <div class="ui-minimize-bar" onclick="toggleUI()" id="ui-toggle-btn">▼ MINIMIZE CONSOLE ▼</div>
                <div class="ui-bottom" id="ui-panel">
                    <div class="search-container">
                        <input type="text" id="query" placeholder="Enter query parameters...">
                        <div id="output">System Ready. Waiting for input...</div>
                        <div id="xray-panel"></div>
                    </div>
                    <div style="min-width: 180px;">
                        <button onclick="search()">EXECUTE</button>
                        <button onclick="toggleXray()">TOGGLE X-RAY</button>
                        <button class="btn-danger" onclick="wipeMemory()">WIPE MEMORY</button>
                    </div>
                </div>
            </div>

            <script>
                var isLoading = false;
                var mapInitialized = false;
                
                function toggleUI() {
                    var panel = document.getElementById('ui-panel');
                    var btn = document.getElementById('ui-toggle-btn');
                    if (panel.classList.contains('ui-hidden')) {
                        panel.classList.remove('ui-hidden');
                        btn.innerText = '▼ MINIMIZE CONSOLE ▼';
                    } else {
                        panel.classList.add('ui-hidden');
                        btn.innerText = '▲ EXPAND CONSOLE ▲';
                    }
                }

                function updateMap() {
                    fetch('/points')
                        .then(res => res.json())
                        .then(d => {
                            if (!d.x || d.x.length === 0) {
                                Plotly.react('chart', [], { scene: { bgcolor: '#05070a' }, paper_bgcolor: '#05070a' });
                                document.getElementById('node-count').innerText = "0";
                                return;
                            }
                            
                            document.getElementById('node-count').innerText = d.total_nodes;
                            document.getElementById('orbit-coords').innerText = `X:${d.ox.toFixed(2)} Y:${d.oy.toFixed(2)} Z:${d.oz.toFixed(2)}`;

                            var trace1 = { x: d.x, y: d.y, z: d.z, text: d.labels, mode: 'markers', type: 'scatter3d', hoverinfo: 'none', marker: {size: 3, color: '#00ff41', opacity: 0.6} };
                            var orbit = { x: [d.ox], y: [d.oy], z: [d.oz], mode: 'markers', type: 'scatter3d', name: 'Center of Gravity', marker: {size: 12, color: '#ff003c', symbol: 'diamond', opacity: 0.9} };
                            
                            Plotly.react('chart', [trace1, orbit], { 
                                uirevision: 'true',
                                scene: { 
                                    bgcolor: '#05070a', 
                                    xaxis: {showgrid: false, zeroline: false, showticklabels: false},
                                    yaxis: {showgrid: false, zeroline: false, showticklabels: false},
                                    zaxis: {showgrid: false, zeroline: false, showticklabels: false}
                                }, 
                                paper_bgcolor: '#05070a', margin: {l:0,r:0,t:0,b:0}, showlegend: false 
                            });

                            // The Star Inspector Feature
                            if (!mapInitialized) {
                                document.getElementById('chart').on('plotly_click', function(data){
                                    if(data.points.length > 0 && data.points[0].text) {
                                        var text = data.points[0].text;
                                        var xray = document.getElementById('xray-panel');
                                        xray.innerHTML = "<span style='color:#00ff41;'><b>[STAR INSPECTOR]</b></span>\\n" + text;
                                        xray.style.display = 'block';
                                        
                                        // Auto-expand UI if it was hidden
                                        var panel = document.getElementById('ui-panel');
                                        if (panel.classList.contains('ui-hidden')) toggleUI();
                                    }
                                });
                                mapInitialized = true;
                            }
                        })
                        .catch(err => console.error('Map Error:', err));
                }
                
                function search() {
                    var queryInput = document.getElementById('query');
                    var q = queryInput.value.trim();
                    if (!q || isLoading) return;
                    
                    isLoading = true;
                    document.getElementById('output').innerHTML = '<span style="color:#ffaa00;">Traversing FAISS Graph & Generating Response...</span>';
                    document.getElementById('status-text').innerText = 'COMPUTING';
                    document.getElementById('status-text').style.color = '#ffaa00';
                    
                    fetch('/search', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text: q}) })
                        .then(res => res.json())
                        .then(d => {
                            document.getElementById('output').innerText = d.answer;
                            
                            var xrayHtml = "<b>RAW RETRIEVED CONTEXT (Cross-Encoder Top-K):</b>\\n\\n";
                            d.context.forEach((chunk, i) => {
                                let highlightedChunk = chunk.replace(/(\\[SOURCE:.*?\\])/g, '<span class="source-tag">$1</span>');
                                xrayHtml += `[Match ${i+1}]\\n${highlightedChunk}\\n\\n`;
                            });
                            document.getElementById('xray-panel').innerHTML = xrayHtml;
                            
                            queryInput.value = '';
                            isLoading = false;
                            document.getElementById('status-text').innerText = 'ONLINE';
                            document.getElementById('status-text').style.color = '#00ff41';
                            updateMap(); 
                        })
                        .catch(err => {
                            document.getElementById('output').innerText = 'System Error: ' + err.message;
                            isLoading = false;
                        });
                }

                function toggleXray() {
                    var panel = document.getElementById('xray-panel');
                    panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
                }

                function wipeMemory() {
                    if(!confirm("WARNING: This will permanently delete the SQLite vault and FAISS graph. Proceed?")) return;
                    
                    fetch('/wipe', { method: 'POST' })
                        .then(res => res.json())
                        .then(d => {
                            document.getElementById('output').innerText = "SYSTEM PURGED: " + d.message;
                            updateMap();
                        });
                }
                
                document.addEventListener('DOMContentLoaded', function() {
                    document.getElementById('query').addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') search();
                    });
                });
                
                setTimeout(updateMap, 500);
                setInterval(updateMap, 10000);
            </script>
        </body>
    </html>
    """

# Create a global cache to stop the map from spinning
MAP_CACHE = {
    "total_nodes": -1,
    "x": [], "y": [], "z": [], "labels": [],
    "pca": None
}

@app.get("/points")
async def get_points():
    try:
        total_nodes = memory.faiss_index.ntotal if memory.faiss_index else 0
        
        # 1. BUILD THE STATIC GALAXY MAP (Only runs if the database size changes)
        if total_nodes > 2 and total_nodes != MAP_CACHE["total_nodes"]:
            sample_size = min(total_nodes, 2000)
            
            memory.cursor.execute(f"SELECT id, text FROM vault ORDER BY RANDOM() LIMIT {sample_size}")
            rows = memory.cursor.fetchall()
            
            vectors = []
            labels = []
            for row in rows:
                idx, text = row
                try:
                    vec = memory.faiss_index.reconstruct(idx)
                    vectors.append(vec)
                    labels.append(text[:80] + "...")
                except:
                    continue

            if len(vectors) > 2:
                combined = np.array(vectors)
                pca = PCA(n_components=3)
                low_dim = pca.fit_transform(combined)
                
                # Save the fixed math to the cache
                MAP_CACHE["pca"] = pca
                MAP_CACHE["x"] = [float(v) for v in low_dim[:, 0]]
                MAP_CACHE["y"] = [float(v) for v in low_dim[:, 1]]
                MAP_CACHE["z"] = [float(v) for v in low_dim[:, 2]]
                MAP_CACHE["labels"] = labels
                MAP_CACHE["total_nodes"] = total_nodes

        # 2. PLACE THE RED DIAMOND (Using the instantaneous search vector!)
        if LATEST_QUERY_VECTOR is not None and MAP_CACHE["pca"] is not None:
            # Map the exact location where the FAISS engine just struck
            query_3d = MAP_CACHE["pca"].transform(LATEST_QUERY_VECTOR)
            ox, oy, oz = float(query_3d[0, 0]), float(query_3d[0, 1]), float(query_3d[0, 2])
        else:
            # If no search has happened yet, put it at 0,0,0
            ox, oy, oz = 0.0, 0.0, 0.0

        return {
            "x": MAP_CACHE["x"], "y": MAP_CACHE["y"], "z": MAP_CACHE["z"],
            "labels": MAP_CACHE["labels"],
            "ox": ox, "oy": oy, "oz": oz,
            "total_nodes": total_nodes
        }
    except Exception as e:
        print(f"Error in /points: {e}")
        return {"x": [], "y": [], "z": [], "labels": [], "ox": 0, "oy": 0, "oz": 0, "total_nodes": 0}
        
class Query(BaseModel): text: str

@app.post("/search")
async def search(q: Query):
    global LATEST_QUERY_VECTOR
    try:
        # INTERCEPT THE EXACT MATH OF THE INSTANTANEOUS SEARCH
        objective_vector = torch.tensor(memory.embedder.encode(q.text)).unsqueeze(0)
        LATEST_QUERY_VECTOR = F.normalize(objective_vector, p=2, dim=1).numpy()
        
        # Recall from Swarm
        resonant_context = memory.recall(q.text, top_k=3)
        context_string = "\n---\n".join(resonant_context)
        
        system_prompt = """You are a strictly constrained data-extraction AI. You have NO external knowledge. You are completely blind to the outside world, coding, pop culture, and general trivia.
        Your ONLY job is to answer the user's question using EXCLUSIVELY the provided DOCUMENT CHUNKS.
        
        CRITICAL RULES:
        1. If the exact answer is not explicitly found in the chunks, you MUST reply with exactly: "I do not have data on this subject in my current memory vault."
        2. Every document chunk provided to you begins with a [SOURCE: ...] tag. If you answer the question, you MUST add a "Sources Used:" list at the very end, citing exactly which tags provided the info.
        """
        
        user_prompt = f"DOCUMENT CHUNKS:\n{context_string}\n\nQUESTION: {q.text}"
        
        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        
        return {
            "answer": response['message']['content'],
            "context": resonant_context
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "context": []}

@app.post("/wipe")
async def wipe_brain():
    global MAP_CACHE, LATEST_QUERY_VECTOR
    try:
        memory.wipe_memory()
        
        # THE FIX: Purge the global cache so the map instantly goes blank
        MAP_CACHE = {
            "total_nodes": -1,
            "x": [], "y": [], "z": [], "labels": [],
            "pca": None
        }
        LATEST_QUERY_VECTOR = None
        
        return {"status": "success", "message": "Memory wiped and cache purged."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)