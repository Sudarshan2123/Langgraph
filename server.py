from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uvicorn
from app import graph

app = FastAPI(title="LangGraph Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    lang: str = "en-US"
    thread_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=request.message, lang=request.lang)]},
            config={"configurable": {"thread_id": request.thread_id}}
        )
        response_text = result["messages"][-1].content
        return ChatResponse(response=response_text, thread_id=request.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
