import os
import chromadb
from fastapi import FastAPI, HTTPException
from groq import Groq

# 1. INITIALIZATION
app = FastAPI(title="SRE Log Analyzer - RAG")

# Connect to Groq (Make sure your API key is in your environment variables)
# If not in env, replace os.environ.get with "your_actual_key"
groq_client = Groq(api_key=Groq_api_key)

# Connect to the Vector DB created in your notebook
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="log_analytics")

# 2. THE SYSTEM PERSONA
# This sets the "Internal Logic" for the AI
SYSTEM_PROMPT = (
    "You are a Senior Site Reliability Engineer (SRE). "
    "Your mission is to perform root cause analysis using only the provided logs. "
    "Rules:\n"
    "1. Be technical and concise.\n"
    "2. If the logs contain an error code (e.g., OOM-kill, 404, 500), highlight it.\n"
    "3. Do not mention that you are an AI; act as a colleague.\n"
    "4. If the answer is not in the logs, say 'ROOT CAUSE NOT FOUND IN CONTEXT'."
)

# 3. THE ENDPOINT
@app.get("/ask")
async def ask_question(question: str):
    try:
        # A. RETRIEVAL: Get the 2 most relevant logs
        results = collection.query(
            query_texts=[question],
            n_results=2
        )
        
        # Merge the retrieved text into one context block
        retrieved_logs = "\n---\n".join(results['documents'][0])
        sources = results['ids'][0]

        # B. THE "MESSAGES" ARGUMENT (Strictly Separated Roles)
        messages = [
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": f"LOG CONTEXT:\n{retrieved_logs}\n\nUSER QUESTION: {question}"
            }
        ]

        # C. GENERATION: Call Groq
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.1 # Low temperature for factual accuracy
        )

        return {
            "analysis": chat_completion.choices[0].message.content,
            "metadata": {
                "sources_queried": sources,
                "model_used": "llama-3.1-8b-instant"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. HEALTH CHECK
@app.get("/")
def root():
    return {"status": "Active", "logs_indexed": collection.count()}