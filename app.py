import os
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import functions from main.py
from main import (
    parse_chat, 
    extract_qa_pairs, 
    build_embedding_index, 
    semantic_search, 
    generate_answer_with_context,
    extract_additional_statements
)
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Chat Parser UI")

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Global variables to store model and data
chat_file = "_chat.txt"
target_answerer = "R Babu"
model = None
texts = None
embeddings = None
qa_pairs = None
messages = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    query: str
    contexts: List[dict]
    answer: str

@app.on_event("startup")
async def startup_event():
    global model, texts, embeddings, qa_pairs, messages
    
    # Load and process chat data
    messages = parse_chat(chat_file)
    print(f"Parsed {len(messages)} messages from chat.")
    
    qa_pairs = extract_qa_pairs(messages, answerer=target_answerer)
    print(f"Extracted {len(qa_pairs)} Q&A pairs.")

    # Or, if you prefer only additional statements not already in Q/A pairs:
    additional_statements = extract_additional_statements(messages, answerer=target_answerer, qa_pairs=qa_pairs)
    print(f"Additional standalone statements by {target_answerer}: {len(additional_statements)}")
    
    # Initialize the semantic model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts, embeddings = build_embedding_index(qa_pairs, additional_statements, model)
    print("Built embedding index for Q&A pairs.")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/query")
async def process_query(query_request: QueryRequest):
    global model, texts, embeddings
    
    if model is None or texts is None or embeddings is None:
        raise HTTPException(status_code=500, detail="Model or data not initialized")
    
    # Retrieve the top relevant Q&A contexts
    retrieved = semantic_search(
        query_request.query, 
        texts, 
        embeddings, 
        model, 
        top_k=query_request.top_k
    )
    
    # Format the results
    contexts = [
        {"text": context, "score": round(score * 100, 2)} 
        for context, score in retrieved
    ]
    
    # Generate a final answer
    answer = generate_answer_with_context(
        query_request.query, 
        [context for context, _ in retrieved], 
        use_openai=False
    )
    
    return QueryResponse(
        query=query_request.query,
        contexts=contexts,
        answer=answer
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 