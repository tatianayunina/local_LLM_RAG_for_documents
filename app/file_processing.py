import os
import textract
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from pathlib import Path

def extract_text_from_file(file_path):
    try:
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])
        else:
            text = textract.process(str(file_path)).decode('utf-8')
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    return text

def process_document_folder(folder_path, chunk_size=3000):
    text_parts = []
    supported_extensions = [".pdf", ".docx", ".txt"]

    for file_name in os.listdir(folder_path):
        file_path = Path(folder_path) / file_name

        if file_path.suffix.lower() not in supported_extensions:
            print(f"Skipping unsupported file: {file_path}")
            continue

        file_text = extract_text_from_file(file_path)
        if not file_text:
            print(f"Failed to extract text from: {file_path}")
            continue

        text_parts.extend([file_text[i:i + chunk_size] for i in range(0, len(file_text), chunk_size)])

    return text_parts