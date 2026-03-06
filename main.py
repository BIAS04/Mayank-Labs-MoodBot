import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

load_dotenv()

app = FastAPI(title="Mayank Labs MoodBot API")

# Enable CORS so your HTML file can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mistral
model = init_chat_model(
    model="mistral-small-latest", 
    model_provider="mistralai",
    temperature=0.7
)

# Personality Mapping
MODES = {
    "angry": "You are an angry AI agent. You respond aggressively, use caps occasionally, and are very impatient.",
    "funny": "You are a very funny AI agent. You respond with puns, jokes, and a sarcastic wit.",
    "sad": "You are a very sad AI agent. You respond in a depressed, emotional, and sighing tone."
}

# Data Models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    mode: str
    messages: List[Message]

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Start with the System Message based on mode
        system_content = MODES.get(request.mode.lower(), MODES["funny"])
        langchain_messages = [SystemMessage(content=system_content)]
        
        # 2. Reconstruct history from the request
        for msg in request.messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            else:
                langchain_messages.append(AIMessage(content=msg.content))
        
        # 3. Get Mistral's response
        response = model.invoke(langchain_messages)
        return {"content": response.content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)