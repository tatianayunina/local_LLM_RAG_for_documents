import requests
import json

def ask_llm(question, context, model="llama3.2"):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": f"Context:\n{context}\n\nQuestion: {question}",
            "max_tokens": 1000
        }
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        final_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    final_response += data.get("response", "")
                    if data.get("done", False):
                        break
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line}, error: {e}")

        return final_response.strip() or "No response from the model."
    except Exception as e:
        print(f"Error while communicating with Ollama: {e}")
        return "Error while generating response."