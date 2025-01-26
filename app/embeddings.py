import os
import pickle
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDINGS_DIR = "app"

def save_embeddings(embeddings, filename="embeddings.pkl"):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    file_path = os.path.join(EMBEDDINGS_DIR, filename)
    with open(file_path, "wb") as file:
        pickle.dump(embeddings, file)
    print(f"Embeddings saved to {file_path}")

def load_embeddings(filename="embeddings.pkl"):
    file_path = os.path.join(EMBEDDINGS_DIR, filename)
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            embeddings = pickle.load(file)
        print(f"Embeddings loaded from {file_path}")
        
        if embeddings is None or len(embeddings) == 0:
            print("Loaded embeddings are empty.")
            return None
        
        return embeddings
    else:
        print(f"{file_path} does not exist. No embeddings to load.")
    return None

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(text_parts, saved_embeddings_file="embeddings.pkl"):
    embeddings = load_embeddings(saved_embeddings_file)
    if embeddings is None:
        embeddings = []
        try:
            embeddings = model.encode(text_parts, convert_to_numpy=True)
        except Exception as e:
            print(f"Error during embedding generation: {e}")
        save_embeddings(embeddings, saved_embeddings_file)
    else:
        print("Loaded existing embeddings.")
    return embeddings