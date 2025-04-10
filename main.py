from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import ChatRequest, ChatResponse
from utils.utils import get_chat_completion
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    response = get_chat_completion(chat_request.query)
    return ChatResponse(
        user_id=chat_request.user_id,
        response=response
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}