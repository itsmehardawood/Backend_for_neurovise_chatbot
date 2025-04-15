from decimal import Decimal
from pydantic import BaseModel
from typing import List, Dict

class ChatRequest(BaseModel):
    user_id: str
    query: str

class ChatResponse(BaseModel):
    user_id: str
    response: str

# ─── MODELS ───────────────────────────────────────────────────────────────────
class User(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str

class WorkingHours(BaseModel):
    start: str
    end: str
    active: bool

class Service(BaseModel):
    serviceName: str
    description: str
    priceType: str
    price: Decimal
    isActive: bool
    id: int
    working_hours: Dict[str, WorkingHours] = None  # Optional working hours per service

class BusinessSettings(BaseModel):
    services: List[Service]
    chat_tone: str  # Removed the top-level working_hours field here