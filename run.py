from app.main import app
from app.main import app, initialize_embeddings_and_index

if __name__ == "__main__":
    initialize_embeddings_and_index()
    app.run(debug=True)