import os
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template, jsonify
from app.file_processing import process_document_folder
from app.embeddings import get_embeddings, save_embeddings, load_embeddings
from app.faiss_index import create_faiss_index, find_similar_parts
from app.llm_ollama import ask_llm

app = Flask(__name__)

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
DOCUMENTS_PATH = project_root / "local_LLM_RAG_for_documents" / "data"

embeddings = None
index = None
text_parts = None

def initialize_embeddings_and_index():
    global embeddings, index, text_parts
    embeddings = load_embeddings()
    if embeddings is not None and embeddings.size > 0:
        index = create_faiss_index(embeddings)
    else:
        print("No embeddings found. FAISS index will be created upon new data.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global embeddings, index, text_parts

    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "The question is mandatory"}), 400

    new_text_parts = process_document_folder(DOCUMENTS_PATH)

    if not new_text_parts:
        return jsonify({"error": "No valid files found in the folder."}), 400

    if text_parts != new_text_parts:
        text_parts = new_text_parts
        new_embeddings = get_embeddings(text_parts)

        if embeddings is not None:
            embeddings = np.concatenate((embeddings, new_embeddings))
        else:
            embeddings = new_embeddings

        save_embeddings(embeddings)

        index = create_faiss_index(embeddings)
    else:
        print("No new data. Using existing embeddings and index.")

    relevant_parts = find_similar_parts(index, user_question, text_parts, top_k=3)
    if not relevant_parts:
        return jsonify({"answer": "Couldn't find relevant parts of the text."})

    context = "\n".join(relevant_parts)

    answer = ask_llm(user_question, context)
    return jsonify({"answer": answer})