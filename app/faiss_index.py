import numpy as np
import faiss
import requests
import os
from sentence_transformers import SentenceTransformer
import numpy as np

def create_faiss_index(embeddings):
    if os.path.exists("faiss_index.index"):
        print("Loading existing FAISS index.")
        index = faiss.read_index("faiss_index.index")
    else:
        embeddings_matrix = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
        index.add(embeddings_matrix)
        faiss.write_index(index, "faiss_index.index")
    return index

model = SentenceTransformer('all-MiniLM-L6-v2')

def find_similar_parts(index, query, text_parts, top_k=3):
    try:
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)

        if len(indices[0]) == 0:
            print("No similar parts found.")
            return []

        return [text_parts[i] for i in indices[0] if i < len(text_parts)]
    except Exception as e:
        print(f"Error in find_similar_parts: {e}")
        return []