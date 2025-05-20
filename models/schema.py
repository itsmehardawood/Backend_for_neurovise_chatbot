from decimal import Decimal
from pydantic import BaseModel
from typing import List, Dict
from typing import Optional

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
    phone: str

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
    isActive: bool
    id: int
    working_hours: Optional[Dict[str, WorkingHours]] = None
    
class WorkingHoursUpdate(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    active: Optional[bool] = None

class ServiceUpdate(BaseModel):
    serviceName: Optional[str] = None
    description: Optional[str] = None
    priceType: Optional[str] = None
    price: Optional[Decimal] = None
    isActive: Optional[bool] = None
    working_hours: Optional[Dict[str, WorkingHoursUpdate]] = None    
    
    
    
    
    

class BusinessSettings(BaseModel):
    services: List[Service]
    chat_tone: str  # Removed the top-level working_hours field here
    system_prompt: Optional[str] = None  # <--- ADD THIS