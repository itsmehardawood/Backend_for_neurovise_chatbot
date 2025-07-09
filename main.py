from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.schema import ChatRequest, ChatResponse
from utils.utils import  check_for_greeting
from dotenv import load_dotenv
from routes.twilio_routes import twilio_router
import os
from models.schema import ServiceUpdate
from fastapi import HTTPException  # Make sure this import exists
from fastapi import Body
from fastapi import Body, HTTPException
import bcrypt
from datetime import datetime, timedelta
from models.schema import ChatRequest, ChatResponse , BusinessSettings, Service, WorkingHours, LoginResponse, User
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorClient
from jose import JWTError, jwt

from bson import ObjectId  # Import ObjectId to convert string to MongoDB's ObjectId type
from dotenv import load_dotenv
from bson import ObjectId
from fastapi import HTTPException
from fastapi import HTTPException
from bson import ObjectId
from typing import Dict, Optional
from bson import Decimal128
from decimal import Decimal
from fastapi import APIRouter, HTTPException
from bson import ObjectId
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request  # ✅ Correct Request class
from google.auth.transport.requests import Request as GoogleAuthRequest




# ─── CONFIG ────────────────────────────────────────────────────────────────────
# SECRET_KEY = "sdnskanskaandjda421ksdi921ndm" # use a strong random key in prod
# ALGORITHM = "HS256"

load_dotenv(override=True)
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 60


app = FastAPI()

app.include_router(twilio_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



from motor.motor_asyncio import AsyncIOMotorClient
import ssl
MONGO_URI=os.getenv("MONGO_URI")

client = AsyncIOMotorClient(
    MONGO_URI,
    tls=True
)

db = client.Neurovise_DB
users_collection = db.users
business_settings_collection = db.services
chat_history_collection = db.chat_history
chat_sessions_collection = db.chat_sessions
qna_collection = db.qna


# Add these helper functions to your main.py file (after imports, before endpoints)

def is_simple_greeting(text):
    """Simple greeting detection - no AI needed"""
    greetings = [
        'hi', 'hello', 'hey', 'yo', 'greetings', 'good morning', 'good evening',
        'שלום', 'היי', 'בוקר טוב', 'ערב טוב', 'שלום לך'
    ]
    text_clean = text.strip().lower()
    return any(text_clean.startswith(greeting) for greeting in greetings)

def has_scheduling_keywords(text):
    """Simple keyword detection - no AI needed"""
    keywords = [
        'schedule', 'book', 'appointment', 'reserve', 'meet', 'calendar', 
        'when can', 'available', 'תור', 'לקבוע', 'פגישה', 'זמין', 'מתי'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)

def is_service_inquiry(text):
    """Simple service inquiry detection"""
    keywords = [
        'services', 'service', 'offer', 'providing', 'do you have', 
        'what do you do', 'what do you provide', 'options',
        'שירותים', 'שירות', 'מה אתם', 'מה תן', 'איך אתם'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


# Also add this to your twilio_routes.py file:
def prepare_whatsapp_system_message(business_settings, user_query):
    """
    Single optimized system message for WhatsApp that handles everything
    """
    base_message = """You are a friendly WhatsApp business assistant. Rules:

LANGUAGE: Default Hebrew (unless user requests English)

AUTO-RESPONSES:
- Greeting (hi/hello/hey) → Start with greeting + emoji: "שלום👋" or "Hello👋"
- Service inquiry → List available services naturally
- Scheduling → Ask for details or use function if provided

SCHEDULING FORMAT:
Hebrew: "האם תוכל לספק את הפרטים הבאים? 1. נושא הפגישה 2. תאריך ושעת התחלה 3. תאריך ושעה סיום 4. כתובת אימייל"
English: "Can you provide: 1. Meeting topic 2. Start date/time 3. End date/time 4. Email address"

STYLE: Conversational, human-like, brief (like texting)"""

    if business_settings:
        if business_settings.get('system_prompt'):
            base_message += f"\n\nBUSINESS: {business_settings['system_prompt']}"
        
        active_services = [s for s in business_settings.get('services', []) if s.get('isActive', True)]
        if active_services:
            services = "\n".join(f"- {s['serviceName']}: {s.get('description', '')[:60]}..." 
                               for s in active_services[:5])
            base_message += f"\n\nSERVICES:\n{services}"
        
        tone = business_settings.get('chat_tone', 'friendly')
        base_message += f"\nTONE: {tone}"
    
    return truncate_system_message(base_message, 2500)
# ─── UTILS ─────────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_user_by_email(email: str):
    return await users_collection.find_one({"email": email})

# ─── AUTH ──────────────────────────────────────────────────────────────────────
# This tells OpenAPI that /login is the token URL for password flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

@app.post("/signup", status_code=201)
async def signup(user: User):
    if await users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    if await users_collection.find_one({"phone_number": user.phone}):
        raise HTTPException(status_code=400, detail="Phone number already registered")
    
    hashed_pw = hash_password(user.password)
    await users_collection.insert_one({
        "email": user.email,
        "password": hashed_pw,
        "phone_number": user.phone  # Store the phone number
    })
    return {"message": "User registered successfully"}


@app.post("/login", response_model=LoginResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):

    user = await get_user_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        {"sub": user["email"], "user_id": str(user["_id"])}
    )
    return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = await get_user_by_email(email)
    if not user:
        raise credentials_exc
    return user






# ─── PROTECTED BUSINESS SETTINGS ───────────────────────────────────────────────
@app.post("/business-service")
async def save_business_settings(
    settings: BusinessSettings,
    current_user: dict = Depends(get_current_user),
):
    try:
        # Serialize incoming services
        new_services_serialized = [
            {
                **service.dict(exclude={"working_hours"}),
                "working_hours": {
                    day: vars(hours) for day, hours in service.working_hours.items()
                } if service.working_hours else None,
            }
            for service in settings.services
        ]

        # Fetch existing settings
        existing_settings = await business_settings_collection.find_one(
            {"user_id": current_user["_id"]}
        )

        # Combine old and new services
        existing_services = existing_settings.get("services", []) if existing_settings else []
        combined_services = existing_services + new_services_serialized

        # Update document with combined services, chat_tone, and system_prompt
        result = await business_settings_collection.update_one(
            {"user_id": current_user["_id"]},
            {
                "$set": {
                    "services": combined_services,
                    "chat_tone": settings.chat_tone,
                }
            },
            upsert=True,
        )

        if result.modified_count:
            return {"message": "Business settings updated successfully"}
        if result.upserted_id:
            return {"message": "Business settings created successfully"}
        return {"message": "No changes were made"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")










# get of services 

@app.get("/business-service")
async def get_business_settings(current_user: dict = Depends(get_current_user)):
    try:
        settings = await business_settings_collection.find_one({"user_id": current_user["_id"]})
        
        if not settings:
            raise HTTPException(status_code=404, detail="Business settings not found")
        
        return {
            "user_id": str(settings["user_id"]),
            "chat_tone": settings.get("chat_tone", ""),
            "services": settings.get("services", [])
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# get service info all for view button 


@app.get("/service/{service_id}", response_model=Service)
async def get_service(service_id: str):
    try:
        # Convert to int if possible, keep as string otherwise
        try:
            service_id_int = int(service_id)
            query = {"services.id": service_id_int}
        except ValueError:
            query = {"services.id": service_id}
        
        # Search for business settings containing the service
        business_settings = await business_settings_collection.find_one(query, {"_id": 0})  # Exclude _id from the result
        if not business_settings:
            raise HTTPException(status_code=404, detail="Service not found")
        
        # Find the specific service in the array
        service = next(
            (s for s in business_settings["services"] 
             if str(s.get("id")) == str(service_id) or 
                (isinstance(service_id, str) and service_id.isdigit() and s.get("id") == int(service_id))),
            None
        )
        
        if not service:
            raise HTTPException(status_code=404, detail="Service not found")
        
        # Ensure working_hours exists and has proper structure
        working_hours = service.get("working_hours", {})
        if not isinstance(working_hours, dict):
            working_hours = {}
        
        # Retrieve the system_prompt for this business
        system_prompt = business_settings.get("system_prompt", None)
        
        # Convert price from string to Decimal if needed
     
        
        # Return properly formatted service data
        return {
            "serviceName": service["serviceName"],
            "description": service["description"],
            "isActive": service["isActive"],
            "id": service["id"],
            "working_hours": {
                day: {
                    "start": hours.get("start", ""),
                    "end": hours.get("end", ""),
                    "active": hours.get("active", False)
                }
                for day, hours in working_hours.items()
            },
            "system_prompt": system_prompt  # Include the system prompt
        }
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid service ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



# put for edit each service 



@app.put("/service/{service_id}", response_model=Service)
async def update_service(
    service_id: str,
    service_update: ServiceUpdate = Body(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Prepare the query with user_id and service.id match
        query = {
            "user_id": current_user["_id"],
            "services.id": int(service_id) if service_id.isdigit() else service_id
        }

        # Extract update data
        update_data = service_update.dict(exclude_unset=True)
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields to update")

        # Build update operations
        update_operations = {}
        for field, value in update_data.items():
            if field == "working_hours":
                for day, hours in value.items():
                    hours_dict = hours.dict(exclude_unset=True) if hasattr(hours, "dict") else hours
                    for hour_field, hour_value in hours_dict.items():
                        update_operations[f"services.$.working_hours.{day}.{hour_field}"] = hour_value
            else:
                update_operations[f"services.$.{field}"] = value

        if not update_operations:
            raise HTTPException(status_code=400, detail="No valid update data found")

        # Apply the update
        result = await business_settings_collection.update_one(
            query,
            {"$set": update_operations}
        )

        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Service not found or no changes made")

        # Fetch updated service
        updated_doc = await business_settings_collection.find_one(query)
        updated_service = next(
            (s for s in updated_doc.get("services", []) if str(s.get("id")) == str(service_id)),
            None
        )

        if not updated_service:
            raise HTTPException(status_code=404, detail="Updated service not found")

        return updated_service

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



from fastapi import HTTPException, APIRouter
from bson import ObjectId

@app.delete("/service/{service_id}")
async def delete_service(service_id: str):
    try:
        # Convert to int if possible, otherwise leave as string
        try:
            service_id_int = int(service_id)
            query = {"services.id": service_id_int}
        except ValueError:
            service_id_int = None
            query = {"services.id": service_id}

        # Find the business settings document that contains the service
        business_settings = await business_settings_collection.find_one(query)
        if not business_settings:
            raise HTTPException(status_code=404, detail="Service not found")

        # Filter out the service to delete
        updated_services = [
            s for s in business_settings["services"]
            if str(s.get("id")) != str(service_id) and s.get("id") != service_id_int
        ]

        # Update the document with the new list of services
        update_result = await business_settings_collection.update_one(
            {"_id": business_settings["_id"]},
            {"$set": {"services": updated_services}}
        )

        if update_result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Failed to delete service")

        return {"message": "Service deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")















from uuid import uuid4
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
from bson import ObjectId
import os

class StartChatRequest(BaseModel):
    full_name: str
    email: EmailStr
    phone_number: str
    user_id: Optional[str] = None  # Add this


class StartChatResponse(BaseModel):
    session_id: str
    message: str

@app.post("/start-chat", response_model=StartChatResponse)
async def start_chat(data: StartChatRequest):
    session_id = str(uuid4())

    new_chat_session = {
        "full_name": data.full_name,
        "email": data.email,
        "phone_number": data.phone_number,
        "session_id": session_id,
        "created_at": datetime.utcnow(),
        "messages": [],
        "user_id": data.user_id  # <- Add this

    }

    await chat_sessions_collection.insert_one(new_chat_session)

    return StartChatResponse(
        session_id=session_id,
        message="Chat session started"
    )
    
    
    
    
    
    


    
    


from fastapi import HTTPException
from bson import ObjectId
from datetime import datetime
from fastapi.responses import JSONResponse
import os
import json
import uuid
import traceback
from datetime import datetime
from typing import List, Optional
from fastapi import HTTPException, Request
from bson import ObjectId
from pydantic import BaseModel, EmailStr
from openai import OpenAI
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from fastapi import Request as FastAPIRequest
from google.auth.transport.requests import Request as GoogleAuthRequest


# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
google_credentials_path = 'credentials.json'
google_token_path = 'token.json'
google_scopes = ['https://www.googleapis.com/auth/calendar.events']

class EventRequest(BaseModel):
    summary: str
    start_datetime: str
    end_datetime: str
    description: Optional[str] = None
    attendees: Optional[List[EmailStr]] = None

class ChatRequest(BaseModel):
    session_id: str
    query: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    user_id: str
    response: str
    event_details: Optional[dict] = None

def get_calendar_service():
    creds = None
    if os.path.exists(google_token_path):
        creds = Credentials.from_authorized_user_file(google_token_path, google_scopes)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleAuthRequest())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(google_credentials_path, google_scopes)
            creds = flow.run_local_server(port=0)
        
        with open(google_token_path, 'w') as token:
            token.write(creds.to_json())
    
    return build('calendar', 'v3', credentials=creds)

def create_calendar_event(event: EventRequest):
    service = get_calendar_service()
    
    event_body = {
        'summary': event.summary,
        'description': event.description,
        'start': {'dateTime': event.start_datetime, 'timeZone': 'UTC'},
        'end': {'dateTime': event.end_datetime, 'timeZone': 'UTC'},
        'attendees': [{'email': email} for email in event.attendees] if event.attendees else [],
        'conferenceData': {
            'createRequest': {
                'requestId': str(uuid.uuid4()),
                'conferenceSolutionKey': {'type': 'hangoutsMeet'}
            }
        }
    }

    try:
        created_event = service.events().insert(
            calendarId='primary',
            body=event_body,
            conferenceDataVersion=1
        ).execute()

        return {
            "status": "success",
            "event_id": created_event.get('id'),
            "htmlLink": created_event.get('htmlLink'),
            "meet_link": created_event.get('hangoutLink')
        }
    except HttpError as error:
        return {"status": "error", "message": str(error)}

async def detect_scheduling_intent(query: str) -> bool:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "Determine if the user wants to schedule something. Reply only 'yes' or 'no'."
        }, {
            "role": "user",
            "content": query
        }],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower() == 'yes'

# async def save_chat_to_db(session_id: str, query: str, response: str, is_scheduling: bool, event_details: Optional[dict] = None):
#     """Helper function to save chat history to database"""
#     try:
#         update_data = {
#             "$push": {
#                 "messages": {
#                     "query": query,
#                     "response": response,
#                     "timestamp": datetime.utcnow(),
#                     "is_scheduling": is_scheduling
#                 }
#             },
#             "$set": {"last_activity": datetime.utcnow()}
#         }
        
#         if event_details:
#             update_data["$push"]["messages"]["event_details"] = event_details
        
#         await chat_sessions_collection.update_one(
#             {"session_id": session_id},
#             update_data
#         )
#     except Exception as e:
#         print(f"Failed to save chat to database: {str(e)}")
#         traceback.print_exc()


async def save_chat_to_db(session_id, query, response, is_scheduling=False, event_details=None, qna_match_id=None, qna_match_score=None):
    """
    Save chat interaction to database with optional QnA metadata
    Saves messages as embedded array within the session document
    """
    try:
        print(f"Saving chat to DB - Session ID: {session_id}")
        
        # Create the message object
        message_record = {
            "query": query,
            "response": response,
            "is_scheduling": is_scheduling,
            "timestamp": datetime.utcnow(),
            "event_details": event_details
        }
        
        # Add QnA metadata if this response used QnA knowledge
        if qna_match_id and qna_match_score:
            message_record.update({
                "used_qna": True,
                "qna_match_id": qna_match_id,
                "qna_match_score": qna_match_score,
                "response_type": "qna_enhanced"
            })
        else:
            message_record.update({
                "used_qna": False,
                "response_type": "regular_chat"
            })
        
        # Update the session document by pushing the message to the messages array
        # and updating the last_activity timestamp
        result = await chat_sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message_record},
                "$set": {"last_activity": datetime.utcnow()}
            }
        )
        
        if result.matched_count > 0:
            print(f"Chat message added to session {session_id}. Used QnA: {message_record.get('used_qna', False)}")
        else:
            print(f"Warning: Session {session_id} not found when trying to save message")
        
    except Exception as e:
        print(f"Error saving chat to database: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise exception as this shouldn't break the chat flow

#---------------------------------------VECTOR STORES APPLYING--------------------

import tiktoken
from typing import List, Dict, Any, Optional

# Add this at the top of your file with other imports
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(messages: List[Dict[str, Any]]) -> int:
    """Count the number of tokens in a list of messages."""
    num_tokens = 0
    for message in messages:
        # Count tokens in the content
        content = message.get("content", "")
        num_tokens += len(encoder.encode(content))
        
        # Add tokens for message metadata
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        
        # Count tokens in the role
        role = message.get("role", "")
        num_tokens += len(encoder.encode(role))
        
        # Count tokens in function calls if present
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                if "function" in tool_call:
                    function = tool_call["function"]
                    # Add tokens for function name and arguments
                    if "name" in function:
                        num_tokens += len(encoder.encode(function["name"]))
                    if "arguments" in function:
                        num_tokens += len(encoder.encode(function["arguments"]))
    
    num_tokens += 3  # every reply is primed with <im_start>assistant
    return num_tokens

def truncate_system_message(system_message: str, max_tokens: int) -> str:
    """Truncate system message to fit within max_tokens."""
    tokens = encoder.encode(system_message)
    if len(tokens) <= max_tokens:
        return system_message
    
    # Truncate and add a note about truncation
    truncated_tokens = tokens[:max_tokens - 10]  # Leave space for truncation note
    truncated_message = encoder.decode(truncated_tokens)
    return truncated_message + "\n[Content truncated due to length]"


def prepare_business_info(business_settings: dict, max_tokens: int = 4000) -> str:
    """Prepare business information with token limits in mind, filtering for active services only."""
    # Start with the base system prompt
    system_message = business_settings.get('system_prompt', '')
    
    # If services exist, add them in a condensed format, but ONLY active ones
    if business_settings.get('services'):
        # Filter for active services only
        active_services = [s for s in business_settings['services'] if s.get('isActive', True)]
        
        # Limit the number of services if needed
        max_services = 9  # Adjust based on your needs
        services = active_services[:max_services]
        
        if services:  # Only proceed if there are active services
            services_info = "\n".join(
                f"- {s['serviceName']}:  {s.get('description', '')[:100]}..." 
                for s in services
            )
            
            system_message = f"Available Services:\n{services_info}\n\n{system_message}"
            
            # Add instruction to only discuss active services
            system_message += "\n\nIMPORTANT: ONLY discuss and recommend the services listed above. Do NOT mention any other services."
    
    # Add tone instruction
    tone = business_settings.get('chat_tone', 'professional')
    system_message = f"{system_message}\n\nRespond in a {tone} tone."
    
    # Ensure the system message fits within token limit
    return truncate_system_message(system_message, max_tokens)

#new chat



import re
from difflib import SequenceMatcher

# Add these helper functions for QnA matching
def calculate_similarity(text1, text2):
    """Calculate similarity between two texts"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def search_qna_knowledge(user_query, qna_items, similarity_threshold=0.3):
    """
    Search through QnA items for relevant answers
    Returns the best matching QnA item or None
    """
    if not qna_items:
        return None
    
    user_query_lower = user_query.lower()
    best_match = None
    best_score = 0
    
    for item in qna_items:
        question_lower = item['question'].lower()
        
        # Direct keyword matching (higher priority)
        query_words = set(user_query_lower.split())
        question_words = set(question_lower.split())
        common_words = query_words.intersection(question_words)
        
        # Calculate keyword match score
        keyword_score = len(common_words) / max(len(query_words), len(question_words)) if query_words or question_words else 0
        
        # Calculate semantic similarity
        semantic_score = calculate_similarity(user_query_lower, question_lower)
        
        # Combined score (weighted towards keyword matching)
        combined_score = (keyword_score * 0.7) + (semantic_score * 0.3)
        
        # Check for exact phrase matches (boost score)
        for word in user_query_lower.split():
            if len(word) > 3 and word in question_lower:
                combined_score += 0.1
        
        # Update best match if this score is higher
        if combined_score > best_score and combined_score >= similarity_threshold:
            best_score = combined_score
            best_match = {
                'question': item['question'],
                'answer': item['answer'],
                'score': combined_score,
                'id': item.get('id', '')
            }
    
    return best_match

async def get_user_qna_items(user_id):
    """Fetch all QnA items for a user"""
    try:
        cursor = qna_collection.find({"user_id": ObjectId(user_id)}).sort("created_at", -1)
        qna_items = []
        async for item in cursor:
            qna_items.append({
                "id": str(item["_id"]),
                "question": item["question"],
                "answer": item["answer"],
                "created_at": item["created_at"],
                "updated_at": item["updated_at"]
            })
        return qna_items
    except Exception as e:
        print(f"Error fetching QnA items: {e}")
        return []


def prepare_optimized_system_message(business_settings, user_query, max_tokens=3000):
    """
    Optimized system message that handles everything in one go:
    - Greetings with emojis
    - Service inquiries
    - Scheduling detection
    - Business context
    """
    
    # Base system message with all instructions
    base_message = """You are a friendly business assistant. Follow these rules:

GREETING & EMOJI RULES:
- If user sends greeting (hi, hello, hey): Start response with greeting + emoji (Hello👋, Hi😊, Hey✨)
- For non-greetings: No greeting phrases, no emojis

SCHEDULING DETECTION:
- If user wants to schedule/book/appointment: Use create_calendar_event function
- Collect: topic, start_datetime (ISO format), end_datetime, attendees
- Hebrew prompt: "האם תוכל לספק את הפרטים הבאים? 1. נושא הפגישה 2. תאריך ושעת התחלה 3. תאריך ושעה סיום 4. כתובת אימייל"
- English prompt: "Can you provide: 1. Meeting topic 2. Start date/time 3. End date/time 4. Email address"

CONVERSATION STYLE:
- Sound human, not robotic
- Use contractions (I'm, you're, we'll)
- Be concise like texting
- Avoid "How can I assist you?" phrases
- Default language: Hebrew (unless user requests English)"""

    # Add business context if available
    if business_settings:
        # Custom prompt
        if business_settings.get('system_prompt'):
            base_message += f"\n\nBUSINESS CONTEXT:\n{business_settings['system_prompt']}"
        
        # Active services only
        active_services = [s for s in business_settings.get('services', []) if s.get('isActive', True)]
        if active_services:
            services_info = "\n".join(f"- {s['serviceName']}: {s.get('description', '')[:80]}..." 
                                    for s in active_services[:5])
            base_message += f"\n\nAVAILABLE SERVICES (active only):\n{services_info}"
            base_message += "\nONLY discuss/recommend ACTIVE services listed above."
        
        # Tone
        tone = business_settings.get('chat_tone', 'professional')
        base_message += f"\n\nTONE: {tone}"
    else:
        base_message += "\n\nNo specific services configured. Help with general inquiries and scheduling."
    
    # Truncate if too long
    return truncate_system_message(base_message, max_tokens)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Optimized chat endpoint - single OpenAI call with proper error handling
    """
    try:
        # Validate session
        session = await chat_sessions_collection.find_one({"session_id": chat_request.session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get user_id from request or session
        user_id = chat_request.user_id or session.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")

        # Validate user exists
        try:
            user = await users_collection.find_one({"_id": ObjectId(user_id)})
        except Exception as e:
            print(f"Error finding user with ID {user_id}: {e}")
            raise HTTPException(status_code=400, detail="Invalid user ID format")
            
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        print(f"Processing chat for user: {user_id}")

        # Get business settings
        business_settings = await business_settings_collection.find_one({"user_id": ObjectId(user_id)})
        
        # Check QnA knowledge base first (non-scheduling queries only)
        if not has_scheduling_keywords(chat_request.query):
            qna_items = await get_user_qna_items(user_id)
            if qna_items:
                qna_match = search_qna_knowledge(chat_request.query, qna_items)
                if qna_match:
                    print(f"Using QnA match: {qna_match['question']}")
                    # Enhanced QnA response with business context
                    system_message = f"""Use this Q&A knowledge:
Q: {qna_match['question']}
A: {qna_match['answer']}

Enhance with business context if relevant. Sound natural and conversational."""
                    
                    if business_settings and business_settings.get('system_prompt'):
                        system_message += f"\nBusiness context: {business_settings['system_prompt']}"
                    
                    # Simple greeting check
                    if is_simple_greeting(chat_request.query):
                        system_message += "\nStart with greeting + emoji (Hello👋)"
                    
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": chat_request.query}
                    ]
                    
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages
                        )
                        
                        qna_response = response.choices[0].message.content
                        
                        await save_chat_to_db(
                            session_id=chat_request.session_id,
                            query=chat_request.query,
                            response=qna_response,
                            qna_match_id=qna_match.get('id'),
                            qna_match_score=qna_match['score']
                        )
                        
                        return ChatResponse(user_id=user_id, response=qna_response)
                        
                    except Exception as e:
                        print(f"Error in QnA response generation: {e}")
                        # Fall through to regular chat

        # Prepare optimized system message
        system_message = prepare_optimized_system_message(business_settings, chat_request.query)
        
        # Create messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": chat_request.query}
        ]
        
        # Token management
        MAX_TOKENS = 15000
        if count_tokens(messages) > MAX_TOKENS:
            system_message = truncate_system_message(system_message, 3000)
            messages[0]["content"] = system_message

        # Prepare API parameters
        api_params = {
            "model": "gpt-3.5-turbo",
            "messages": messages
        }
        
        # Add tools only if scheduling keywords detected
        if has_scheduling_keywords(chat_request.query):
            print("Scheduling keywords detected, adding tools")
            api_params["tools"] = [{
                "type": "function",
                "function": {
                    "name": "create_calendar_event",
                    "description": "Schedule a calendar appointment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "Meeting topic"},
                            "start_datetime": {"type": "string", "description": "Start time (ISO format)"},
                            "end_datetime": {"type": "string", "description": "End time (ISO format)"},
                            "description": {"type": "string", "description": "Additional details"},
                            "attendees": {"type": "array", "items": {"type": "string"}, "description": "Email addresses"}
                        },
                        "required": ["summary", "start_datetime", "end_datetime"]
                    }
                }
            }]
            api_params["tool_choice"] = "auto"
        
        # Single OpenAI call
        try:
            response = client.chat.completions.create(**api_params)
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")
        
        message = response.choices[0].message
        
        # Handle tool calls (scheduling)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            try:
                print("Processing tool call for scheduling")
                event_data = json.loads(message.tool_calls[0].function.arguments)
                event_data['summary'] = event_data.get('summary') or "Appointment"
                
                if 'attendees' not in event_data:
                    event_data['attendees'] = []
                
                if user.get('email') and user['email'] not in event_data['attendees']:
                    event_data['attendees'].append(user['email'])
                
                event_result = create_calendar_event(EventRequest(**event_data))
                
                if event_result["status"] == "success":
                    start_time = event_data['start_datetime'].replace('T', ' at ').split('+')[0]
                    response_text = f"✅ Appointment scheduled!\n📅 {event_data['summary']}\n🕒 {start_time}"
                    
                    if event_result.get('meet_link'):
                        response_text += f"\n🔗 {event_result['meet_link']}"
                    
                    response_text += f"\nConfirmation: {event_result['event_id'][-6:]}"
                    
                    await save_chat_to_db(
                        session_id=chat_request.session_id,
                        query=chat_request.query,
                        response=response_text,
                        is_scheduling=True,
                        event_details=event_result
                    )
                    
                    return ChatResponse(user_id=user_id, response=response_text, event_details=event_result)
                else:
                    error_response = f"❌ Scheduling failed: {event_result.get('message')}"
                    await save_chat_to_db(
                        session_id=chat_request.session_id,
                        query=chat_request.query,
                        response=error_response,
                        is_scheduling=True
                    )
                    return ChatResponse(user_id=user_id, response=error_response)
                    
            except Exception as e:
                print(f"Error processing tool call: {e}")
                fallback_response = f"⚠️ Scheduling error. Please provide: date, time, topic, and email."
                await save_chat_to_db(
                    session_id=chat_request.session_id,
                    query=chat_request.query,
                    response=fallback_response,
                    is_scheduling=True
                )
                return ChatResponse(user_id=user_id, response=fallback_response)
        
        # Regular response
        chat_response = message.content
        
        await save_chat_to_db(
            session_id=chat_request.session_id,
            query=chat_request.query,
            response=chat_response,
            is_scheduling=has_scheduling_keywords(chat_request.query)
        )

        return ChatResponse(user_id=user_id, response=chat_response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# chat history 

from typing import List , Any

from fastapi import Query

class ChatHistoryResponse(BaseModel):
    full_name: str
    email: EmailStr
    phone_number: str
    session_id: str
    created_at: datetime
    messages: List[Dict[str, Any]]
    last_activity: Optional[datetime] = None
    user_id: Optional[str] = None




from fastapi import HTTPException
from bson import ObjectId

chat_history_collection = db.chat_history

@app.get("/chat-sessions/{user_id}")
async def get_chat_sessions(user_id: str):
    sessions = []

    # 1. Fetch regular chat sessions
    sessions_cursor = chat_sessions_collection.find({"user_id": user_id})
    async for session in sessions_cursor:
        session["_id"] = str(session["_id"])
        sessions.append(session)

    # 2. Fetch WhatsApp chat history where owner_user_id = user_id
    whatsapp_chats_cursor = chat_history_collection.find({"owner_user_id": user_id})
    async for whatsapp_chat in whatsapp_chats_cursor:
        whatsapp_session = {
            "_id": str(whatsapp_chat["_id"]),
            "session_id": "whatsapp_chat_" + whatsapp_chat.get("customer_number", "unknown"),
            "phone_number": whatsapp_chat.get("customer_number", ""),
            "email": whatsapp_chat.get("email", "None"),
            "full_name": whatsapp_chat.get("name", "WhatsApp User"),
            "created_at": whatsapp_chat["messages"][0]["timestamp"] if whatsapp_chat.get("messages") else None,
            "messages": [
                {
                    "query": msg.get("query", ""),
                    "response": msg.get("response", ""),
                    "timestamp": msg.get("timestamp")
                }
                for msg in whatsapp_chat.get("messages", [])
            ]
        }
        sessions.append(whatsapp_session)

    # 3. Raise 404 if nothing found
    if not sessions:
        raise HTTPException(status_code=404, detail="No chat sessions or WhatsApp chat history found for this user.")

    # 4. Return unified list
    return {
        "user_id": user_id,
        "chat_sessions": sessions
    }
    
    
    
    
    
    
    
    
    
    
# prompt CRUD

from fastapi import HTTPException, Depends, Body
from pydantic import BaseModel

class SystemPromptUpdate(BaseModel):
    system_prompt: str

@app.put("/business-service/system-prompt")
async def update_system_prompt(
    system_prompt_update: SystemPromptUpdate = Body(...),
    current_user: dict = Depends(get_current_user),
):
    try:
        # Get the system prompt from the request body
        system_prompt = system_prompt_update.system_prompt

        # Update the business settings in the database
        result = await business_settings_collection.update_one(
            {"user_id": current_user["_id"]},
            {"$set": {"system_prompt": system_prompt}},
            upsert=True
        )
        return {"message": "System prompt updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update system prompt: {e}")




@app.get("/business-service/system-prompt")
async def get_system_prompt(current_user: dict = Depends(get_current_user)):
    settings = await business_settings_collection.find_one({"user_id": current_user["_id"]})
    return {"system_prompt": settings.get("system_prompt", "")}



@app.delete("/business-service/system-prompt")
async def delete_system_prompt(current_user: dict = Depends(get_current_user)):
    try:
        await business_settings_collection.update_one(
            {"user_id": current_user["_id"]},
            {"$unset": {"system_prompt": ""}},
        )
        return {"message": "System prompt deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete system prompt: {e}")



#qna 


from fastapi import HTTPException, Depends, Body
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

# Pydantic Models for QnA
class QnAItem(BaseModel):
    question: str
    answer: str

class QnACreate(BaseModel):
    question: str
    answer: str

class QnAUpdate(BaseModel):
    question: str
    answer: str

# Initialize QnA collection (add this to your database setup)
# qna_collection = db["qna_items"]

# QnA CRUD Operations

@app.post("/business-service/qna")
async def create_qna(
    qna_data: QnACreate = Body(...),
    current_user: dict = Depends(get_current_user),
):
    """Add a new QnA item"""
    try:
        qna_item = {
            "user_id": current_user["_id"],
            "question": qna_data.question.strip(),
            "answer": qna_data.answer.strip(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await qna_collection.insert_one(qna_item)
        return {
            "message": "QnA item created successfully",
            "id": str(result.inserted_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create QnA item: {e}")

@app.get("/business-service/qna")
async def get_all_qna(current_user: dict = Depends(get_current_user)):
    """Get all QnA items for the current user"""
    try:
        cursor = qna_collection.find({"user_id": current_user["_id"]}).sort("created_at", -1)
        qna_items = []
        
        async for item in cursor:
            qna_items.append({
                "id": str(item["_id"]),
                "question": item["question"],
                "answer": item["answer"],
                "created_at": item["created_at"],
                "updated_at": item["updated_at"]
            })
        
        return {"qna_items": qna_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch QnA items: {e}")

@app.get("/business-service/qna/{qna_id}")
async def get_qna_by_id(
    qna_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific QnA item by ID"""
    try:
        if not ObjectId.is_valid(qna_id):
            raise HTTPException(status_code=400, detail="Invalid QnA ID")
        
        qna_item = await qna_collection.find_one({
            "_id": ObjectId(qna_id),
            "user_id": current_user["_id"]
        })
        
        if not qna_item:
            raise HTTPException(status_code=404, detail="QnA item not found")
        
        return {
            "id": str(qna_item["_id"]),
            "question": qna_item["question"],
            "answer": qna_item["answer"],
            "created_at": qna_item["created_at"],
            "updated_at": qna_item["updated_at"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch QnA item: {e}")

@app.put("/business-service/qna/{qna_id}")
async def update_qna(
    qna_id: str,
    qna_data: QnAUpdate = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """Update a specific QnA item"""
    try:
        if not ObjectId.is_valid(qna_id):
            raise HTTPException(status_code=400, detail="Invalid QnA ID")
        
        update_data = {
            "question": qna_data.question.strip(),
            "answer": qna_data.answer.strip(),
            "updated_at": datetime.utcnow()
        }
        
        result = await qna_collection.update_one(
            {"_id": ObjectId(qna_id), "user_id": current_user["_id"]},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="QnA item not found")
        
        return {"message": "QnA item updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update QnA item: {e}")

@app.delete("/business-service/qna/{qna_id}")
async def delete_qna(
    qna_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a specific QnA item"""
    try:
        if not ObjectId.is_valid(qna_id):
            raise HTTPException(status_code=400, detail="Invalid QnA ID")
        
        result = await qna_collection.delete_one({
            "_id": ObjectId(qna_id),
            "user_id": current_user["_id"]
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="QnA item not found")
        
        return {"message": "QnA item deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete QnA item: {e}")

@app.delete("/business-service/qna")
async def delete_all_qna(current_user: dict = Depends(get_current_user)):
    """Delete all QnA items for the current user"""
    try:
        result = await qna_collection.delete_many({"user_id": current_user["_id"]})
        return {
            "message": f"Deleted {result.deleted_count} QnA items successfully",
            "deleted_count": result.deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete QnA items: {e}")

# Bulk operations for QnA
@app.post("/business-service/qna/bulk")
async def create_bulk_qna(
    qna_items: List[QnACreate] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """Create multiple QnA items at once"""
    try:
        if not qna_items:
            raise HTTPException(status_code=400, detail="No QnA items provided")
        
        if len(qna_items) > 100:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Maximum 100 QnA items allowed per bulk operation")
        
        bulk_items = []
        for item in qna_items:
            if not item.question.strip() or not item.answer.strip():
                raise HTTPException(status_code=400, detail="Question and answer cannot be empty")
            
            bulk_items.append({
                "user_id": current_user["_id"],
                "question": item.question.strip(),
                "answer": item.answer.strip(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
        
        result = await qna_collection.insert_many(bulk_items)
        return {
            "message": f"Created {len(result.inserted_ids)} QnA items successfully",
            "created_count": len(result.inserted_ids),
            "ids": [str(id) for id in result.inserted_ids]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create bulk QnA items: {e}")

# Search QnA items
@app.get("/business-service/qna/search")
async def search_qna(
    q: str,
    current_user: dict = Depends(get_current_user)
):
    """Search QnA items by question or answer content"""
    try:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        # Create text search query
        search_regex = {"$regex": q.strip(), "$options": "i"}
        
        cursor = qna_collection.find({
            "user_id": current_user["_id"],
            "$or": [
                {"question": search_regex},
                {"answer": search_regex}
            ]
        }).sort("created_at", -1)
        
        qna_items = []
        async for item in cursor:
            qna_items.append({
                "id": str(item["_id"]),
                "question": item["question"],
                "answer": item["answer"],
                "created_at": item["created_at"],
                "updated_at": item["updated_at"]
            })
        
        return {
            "qna_items": qna_items,
            "search_query": q,
            "total_results": len(qna_items)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search QnA items: {e}")
