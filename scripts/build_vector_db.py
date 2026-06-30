import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/knowledge_base/facts.txt"))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../models/faiss_index"))
print("Starting FAISS Vector Database Construction...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Reading facts from: {KNOWLEDGE_BASE_FILE}")
try:
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as file:
        text_chunks = [line.strip() for line in file.readlines() if line.strip()]
except FileNotFoundError:
    print(f"ERROR: Could not find {KNOWLEDGE_BASE_FILE}.")
    print("Please create this file and add some text facts to it.")
    exit()
print(f"Found {len(text_chunks)} facts to process.")
print("Loading the Sentence Transformer model... (This downloads a small model the first time)")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Converting text into mathematical vectors...")
embeddings = embedder.encode(text_chunks, show_progress_bar=True)

print("Building the FAISS search index...")
vector_dimension = embeddings.shape[1] 
index = faiss.IndexFlatL2(vector_dimension)
index.add(embeddings)

index_path = os.path.join(OUTPUT_DIR, "vector_db.index")
chunks_path = os.path.join(OUTPUT_DIR, "text_chunks.pkl")
faiss.write_index(index, index_path)
with open(chunks_path, "wb") as f:
    pickle.dump(text_chunks, f)

print("\nSuccess!")
print(f"Database saved to: {index_path}")
print(f"Text mappings saved to: {chunks_path}")
print("Your AI now has a long-term memory ready to be searched!")