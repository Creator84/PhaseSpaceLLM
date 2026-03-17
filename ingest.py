import os
import PyPDF2
import ollama
import time
from src.os2_wrapper import PhaseSpaceMemoryNode

print("=== OS² MASSIVE DOCUMENT INGESTOR ===")

def extract_text(file_path):
    """Reads raw text from PDFs and TXT files."""
    text = ""
    if file_path.lower().endswith('.pdf'):
        print(f"[*] Extracting PDF: {file_path}")
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif file_path.lower().endswith('.txt'):
        print(f"[*] Extracting TXT: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file type! Please provide a .txt or .pdf file.")
    return text

def chunk_document(text, chunk_char_limit=800):
    """Chops massive text into model-friendly paragraphs."""
    paragraphs = text.replace('\r', '').split('\n\n')
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        p = p.replace('\n', ' ').strip()
        if not p: continue
        
        if len(current_chunk) + len(p) < chunk_char_limit:
            current_chunk += p + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

if __name__ == "__main__":
    # Put a real PDF or TXT file in your folder and change this name.
    TARGET_FILE = "examples/test.txt" 
    
    if not os.path.exists(TARGET_FILE):
        print(f"[!] {TARGET_FILE} not found. Creating a test document...")
        with open(TARGET_FILE, "w", encoding="utf-8") as f:
            f.write("Grimstad Municipality is located in Agder county, Norway.\n\n")
            f.write("The municipality is known for its maritime history and the University of Agder campus.\n\n")
            f.write("In 2026, the local tech sector saw a massive boom due to local AI initiatives.\n\n")
            f.write("The CEO's favorite food remains pizza, despite the new seafood restaurants opening downtown.")

    raw_text = extract_text(TARGET_FILE)
    document_chunks = chunk_document(raw_text)
    print(f"[*] Sliced document into {len(document_chunks)} mathematical chunks.")

    agent_memory = PhaseSpaceMemoryNode()
    
    start_time = time.time()
    agent_memory.batch_learn(document_chunks)
    end_time = time.time()
    print(f"[*] Successfully mapped document to Phase-Space Orbit in {end_time - start_time:.2f} seconds.")

    print("\n" + "="*40)
    print("READY FOR INTERROGATION")
    print("="*40)
    
    while True:
        try:
            user_query = input("\nAsk the Swarm a question about the document (or type 'quit'): ")
        except KeyboardInterrupt:
            print("\n\n[*] Shutting down the Swarm... Goodbye!")
            break
            
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("[*] Exiting the Swarm interrogation.")
            break
            
        resonant_context = agent_memory.recall(objective=user_query, top_k=2)
        
        context_string = "\n".join(resonant_context)
        prompt = f"""
        You are an analytical AI. Use ONLY the following retrieved document chunks to answer the user's question. 
        If the answer is not in the chunks, say "The document does not specify."
        
        DOCUMENT CHUNKS:
        {context_string}
        
        QUESTION: {user_query}
        """
        
        print("    -> Llama 3.2 is analyzing the math...")
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
        print(f"\n🤖 SWARM ANSWER:\n{response['message']['content']}")
