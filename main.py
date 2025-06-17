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

# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(chat_request: ChatRequest):
#     try:
#         # Validate session and user
#         session = await chat_sessions_collection.find_one({"session_id": chat_request.session_id})
#         if not session:
#             raise HTTPException(status_code=404, detail="Session not found")
        
#         user_id = chat_request.user_id or session.get("user_id")
#         if not user_id:
#             raise HTTPException(status_code=400, detail="User ID required")

#         user = await users_collection.find_one({"_id": ObjectId(user_id)})
#         if not user:
#             raise HTTPException(status_code=404, detail="User not found")

#         # Get business settings
#         business_settings = await business_settings_collection.find_one({"user_id": ObjectId(user_id)})
        
#         # DEBUG: Log user query
#         print(f"Processing chat request. Query: {chat_request.query}")
        
#         # Filter for active services
#         active_services = []
#         if business_settings and business_settings.get('services'):
#             active_services = [s for s in business_settings['services'] if s.get('isActive', True)]
#             print(f"Found {len(active_services)} active services out of {len(business_settings.get('services', []))} total services")
        
#         # Prepare system message - prioritize business settings
#         if business_settings:
#             # Add debugging info about business settings
#             print(f"Found business settings: {business_settings.get('name', 'Unnamed Business')}")
            
#             # Use the token-aware function to prepare business info
#             system_message = prepare_business_info(business_settings)
            
#             # Enhanced system prompt for better appointment handling
#             system_message += "\n\nIMPORTANT: You are a virtual assistant that helps with booking appointments. If a user wants to schedule an appointment, help them by collecting the necessary information and use the provided function to create a calendar event. ONLY discuss and recommend ACTIVE services listed above."
#         else:
#             # Default system message when no business settings exist
#             system_message = """You are a helpful assistant that manages appointments. 
# For appointment scheduling, please collect:
# Can you provide the following details?
# 1. Meeting topic
# 2. Start date and time
# 3. End date and time
# 4. email address

# if user talking in hebrew ask him: 

# האם תוכל לספק את הפרטים הבאים?
# 1. נושא הפגישה
# 2. תאריך ושעת התחלה
# 3. תאריך ושעה סיום
# 4. כתובת אימייל








# IMPORTANT: You CAN and SHOULD book appointments when requested. Use the appointment scheduling function when appropriate.
# Respond in a professional tone."""

#         # Set max tokens limit with buffer
#         MAX_TOKENS = 16000  # Lower than the actual limit of 16385
        
#         # DEBUG: Log system message beginning
#         print(f"System message starts with: {system_message[:100]}...")
        
#         # Determine intent and process
#         is_scheduling = await detect_scheduling_intent(chat_request.query)
        
#         # DEBUG: Log scheduling intent
#         print(f"Scheduling intent detected: {is_scheduling}")
        
#         if is_scheduling:
#             # DEBUG: Log entering scheduling flow
#             print("Entering scheduling flow...")
            
#             # Enhanced system message specific for scheduling
#             scheduling_system_message = system_message
            
#             # Add active services info explicitly for scheduling
#             if active_services:
#                 active_services_info = "\n".join(
#                     f"- {s['serviceName']}: {s.get('description', '')}" 
#                     for s in active_services[:9]  # Limit to 5 services
#                 )
#                 scheduling_system_message += f"\n\nONLY offer these active services for booking:\n{active_services_info}"
#             else:
#                 scheduling_system_message += "\n\nThere are no specific services defined. Schedule a general appointment."
            
#             scheduling_system_message += "\n\nThe user is trying to schedule an appointment. Help them by collecting all necessary information and use the create_calendar_event function to book it. DO NOT refuse to book appointments - that is your primary function. ONLY discuss and recommend ACTIVE services."
            
#             # Define the function call with calendar event schema
#             calendar_function = {
#                 "type": "function",
#                 "function": {
#                     "name": "create_calendar_event",
#                     "description": "Schedule a calendar appointment",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "summary": {"type": "string", "description": "The topic of the meeting"},
#                             "start_datetime": {"type": "string", "description": "Start time in ISO format (YYYY-MM-DDTHH:MM:SS)"},
#                             "end_datetime": {"type": "string", "description": "End time in ISO format (YYYY-MM-DDTHH:MM:SS)"},
#                             "description": {"type": "string", "description": "Additional details about the appointment"},
#                             "attendees": {
#                                 "type": "array",
#                                 "items": {"type": "string", "format": "email"},
#                                 "description": "List of email addresses for attendees"
#                             }
#                         },
#                         "required": ["summary", "start_datetime", "end_datetime"]
#                     }
#                 }
#             }
            
#             # Create messages array
#             messages = [
#                 {"role": "system", "content": scheduling_system_message},
#                 {"role": "user", "content": chat_request.query}
#             ]
            
#             # Check if token count exceeds limit
#             message_tokens = count_tokens(messages)
#             if message_tokens > MAX_TOKENS:
#                 # Calculate how much we need to reduce
#                 excess_tokens = message_tokens - MAX_TOKENS + 500  # Buffer
                
#                 # Recalculate with a shorter system message
#                 max_system_tokens = len(encoder.encode(scheduling_system_message)) - excess_tokens
#                 if max_system_tokens < 500:  # Minimum viable system message
#                     max_system_tokens = 500
                
#                 scheduling_system_message = truncate_system_message(scheduling_system_message, max_system_tokens)
#                 messages[0]["content"] = scheduling_system_message
                
#                 # Verify we're now within limits
#                 if count_tokens(messages) > MAX_TOKENS:
#                     # As a last resort, truncate the user query
#                     user_tokens = len(encoder.encode(chat_request.query))
#                     if user_tokens > 1000:  # Only truncate if it's long
#                         max_query_tokens = user_tokens - (count_tokens(messages) - MAX_TOKENS) - 100
#                         truncated_query = encoder.decode(encoder.encode(chat_request.query)[:max_query_tokens])
#                         truncated_query += " [message truncated]"
#                         messages[1]["content"] = truncated_query
            
#             # DEBUG: Log messages before API call
#             print(f"Making scheduling API call with message tokens: {count_tokens(messages)}")
            
#             # Make the API call with token-managed messages
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=messages,
#                 tools=[calendar_function],
#                 tool_choice="auto"  # Explicitly tell the model to use tools when appropriate
#             )

#             # DEBUG: Log basic response info
#             print(f"Received response. Contains tool calls: {hasattr(response.choices[0].message, 'tool_calls') and bool(response.choices[0].message.tool_calls)}")
            
#             message = response.choices[0].message
            
#             # Check for tool calls from the API response
#             if message.tool_calls:
#                 try:
#                     # DEBUG: Log tool call details
#                     print(f"Tool call received: {message.tool_calls[0].function.name}")
                    
#                     event_data = json.loads(message.tool_calls[0].function.arguments)
#                     print(f"Event data: {event_data}")
                    
#                     # Validate that the requested service is active (if service is specified)
#                     is_valid_service = True
#                     if event_data.get('description') and business_settings and business_settings.get('services'):
#                         # Basic check if the description contains an inactive service name
#                         inactive_service_names = [s['serviceName'].lower() for s in business_settings.get('services', []) 
#                                             if not s.get('isActive', True)]
                        
#                         for inactive_service in inactive_service_names:
#                             if inactive_service in event_data.get('description', '').lower():
#                                 is_valid_service = False
#                                 break
                    
#                     if not is_valid_service:
#                         # If inactive service detected, return a helpful message
#                         inactive_response = "I notice you're interested in a service that isn't currently available. Here are the services we currently offer:\n\n"
#                         inactive_response += "\n".join(
#                             f"- {s['serviceName']}: {s.get('description', '')}" 
#                             for s in active_services
#                         )
#                         inactive_response += "\n\nWould you like to book an appointment for one of these services instead?"
                        
#                         await save_chat_to_db(
#                             session_id=chat_request.session_id,
#                             query=chat_request.query,
#                             response=inactive_response,
#                             is_scheduling=True
#                         )
                        
#                         return ChatResponse(
#                             user_id=user_id,
#                             response=inactive_response
#                         )
                    
#                     event_data['summary'] = event_data.get('summary') or "Appointment"
                    
#                     # Make sure we have both attendees and user email if available
#                     if 'attendees' not in event_data:
#                         event_data['attendees'] = []
                    
#                     if user.get('email') and user['email'] not in event_data['attendees']:
#                         event_data['attendees'].append(user['email'])
                    
#                     # Create the calendar event
#                     event_result = create_calendar_event(EventRequest(**event_data))
                    
#                     if event_result["status"] == "success":
#                         # Format a friendly response for successful scheduling
#                         start_datetime = event_data['start_datetime'].replace('T', ' at ').split('+')[0]
                        
#                         response_text = f"Great! I've scheduled your appointment:\n\n"
#                         response_text += f"📅 {event_data['summary']}\n"
#                         response_text += f"🕒 {start_datetime}\n"
                        
#                         if event_data.get('description'):
#                             response_text += f"📝 {event_data['description']}\n"
                        
#                         if event_result.get('meet_link'):
#                             response_text += f"\n🔗 Video call link: {event_result['meet_link']}"
                        
#                         # Add confirmation number
#                         response_text += f"\n\nConfirmation #: {event_result['event_id'][-6:]}"
                        
#                         # Save to database before returning
#                         await save_chat_to_db(
#                             session_id=chat_request.session_id,
#                             query=chat_request.query,
#                             response=response_text,
#                             is_scheduling=True,
#                             event_details=event_result
#                         )
                        
#                         return ChatResponse(
#                             user_id=user_id,
#                             response=response_text,
#                             event_details=event_result
#                         )
#                     else:
#                         error_response = f"I tried to schedule your appointment, but encountered an error: {event_result.get('message')}"
#                         await save_chat_to_db(
#                             session_id=chat_request.session_id,
#                             query=chat_request.query,
#                             response=error_response,
#                             is_scheduling=True
#                         )
#                         return ChatResponse(
#                             user_id=user_id,
#                             response=error_response
#                         )
#                 except Exception as tool_error:
#                     # Log the specific error with the tool call
#                     print(f"Error processing tool call: {str(tool_error)}")
#                     traceback.print_exc()
                    
#                     # Fall back to regular response
#                     fallback_response = f"I encountered an issue while trying to schedule your appointment: {str(tool_error)}. Please try again with complete details including date, time, and purpose."
                    
#                     await save_chat_to_db(
#                         session_id=chat_request.session_id,
#                         query=chat_request.query,
#                         response=fallback_response,
#                         is_scheduling=True
#                     )
                    
#                     return ChatResponse(
#                         user_id=user_id,
#                         response=fallback_response
#                     )
#             else:
#                 # The model didn't use the function, but we know it's a scheduling intent
#                 # Let's get more information and proceed with normal chat flow
#                 print("Model didn't use tool call despite scheduling intent. Using regular chat response.")
                
#                 # If the API didn't use the tool, we'll use a modified system message
#                 # that encourages getting the details for next time
#                 assist_system_message = system_message + "\n\nThe user wants to schedule an appointment. If you don't have enough details yet, ask Can I help you with a specific topic? like date, time, and purpose. DO NOT refuse to help with scheduling - that is your primary purpose. Never say you cannot book appointments. ONLY mention active services."
                
#                 if active_services:
#                     active_services_info = "\n".join(
#                         f"- {s['serviceName']}: {s.get('description', '')}" 
#                         for s in active_services[:5]
#                     )
#                     assist_system_message += f"\n\nACTIVE SERVICES:\n{active_services_info}"
                
#                 messages = [
#                     {"role": "system", "content": assist_system_message},
#                     {"role": "user", "content": chat_request.query}
#                 ]
                
#                 # Check token count for regular messages
#                 if count_tokens(messages) > MAX_TOKENS:
#                     # Apply token reduction as before
#                     assist_system_message = truncate_system_message(assist_system_message, 4000)
#                     messages[0]["content"] = assist_system_message
                
#                 # Use regular chat completion to ask for more details
#                 chat_completion = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=messages
#                 )
                
#                 # Get text response
#                 chat_response = chat_completion.choices[0].message.content
                
#                 # Apply greeting formatting using the functions from paste-2.txt
        
                
#                 # Save to database
#                 await save_chat_to_db(
#                     session_id=chat_request.session_id,
#                     query=chat_request.query,
#                     response=chat_response,
#                     is_scheduling=True
#                 )
                
#                 return ChatResponse(user_id=user_id, response=chat_response)
        
#         # Regular chat flow (non-scheduling intent)
#         print("Using regular chat flow...")
        
#         # Check if we should add greeting formatting
#         has_greeting = check_for_greeting(chat_request.query)
        
#         # Add greeting instructions if needed
#         if has_greeting:
#             system_message += "\n\nIMPORTANT: Begin your response with a friendly greeting like 'Hello' or 'Hi there' and include an emoji right after the greeting word (with no space). For example: 'Hello👋' or 'Hi there😊'. The rest of your response should NOT contain any emojis."
#         else:
#             system_message += "\n\nIMPORTANT: DO NOT begin your message with greeting phrases like 'hello there', 'hi there', etc. DO NOT use emojis in your response."
        
#         # Create messages for regular chat flow
#         messages = [
#             {"role": "system", "content": system_message},
#             {"role": "user", "content": chat_request.query}
#         ]
        
#         # Check token count for regular messages
#         message_tokens = count_tokens(messages)
#         if message_tokens > MAX_TOKENS:
#             # Apply same token reduction logic as before
#             excess_tokens = message_tokens - MAX_TOKENS + 500  # Buffer
#             max_system_tokens = len(encoder.encode(system_message)) - excess_tokens
#             if max_system_tokens < 500:
#                 max_system_tokens = 500
            
#             system_message = truncate_system_message(system_message, max_system_tokens)
#             messages[0]["content"] = system_message
            
#             # If still too large, truncate user query as last resort
#             if count_tokens(messages) > MAX_TOKENS:
#                 user_tokens = len(encoder.encode(chat_request.query))
#                 if user_tokens > 1000:
#                     max_query_tokens = user_tokens - (count_tokens(messages) - MAX_TOKENS) - 100
#                     truncated_query = encoder.decode(encoder.encode(chat_request.query)[:max_query_tokens])
#                     truncated_query += " [message truncated]"
#                     messages[1]["content"] = truncated_query
        
#         # Make regular chat API call
#         chat_completion = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=messages
#         )
        
#         # Get the response content
#         chat_response = chat_completion.choices[0].message.content
        
        
#         # Save to database
#         await save_chat_to_db(
#             session_id=chat_request.session_id,
#             query=chat_request.query,
#             response=chat_response,
#             is_scheduling=is_scheduling
#         )

#         return ChatResponse(user_id=user_id, response=chat_response)

#     except Exception as e:
#         print(f"Error in chat endpoint: {str(e)}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))




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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        # Validate session and user
        session = await chat_sessions_collection.find_one({"session_id": chat_request.session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        user_id = chat_request.user_id or session.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")

        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get business settings
        business_settings = await business_settings_collection.find_one({"user_id": ObjectId(user_id)})
        
        # DEBUG: Log user query
        print(f"Processing chat request. Query: {chat_request.query}")
        
        # ===== NEW QnA INTEGRATION =====
        # Step 1: Check QnA knowledge base first (unless it's a scheduling intent)
        is_scheduling = await detect_scheduling_intent(chat_request.query)
        
        if not is_scheduling:
            print("Checking QnA knowledge base...")
            
            # Fetch user's QnA items
            qna_items = await get_user_qna_items(user_id)
            
            if qna_items:
                print(f"Found {len(qna_items)} QnA items to search through")
                
                # Search for relevant QnA
                qna_match = search_qna_knowledge(chat_request.query, qna_items)
                
                if qna_match:
                    print(f"Found QnA match with score: {qna_match['score']:.2f}")
                    print(f"Matched question: {qna_match['question']}")
                    
                    # Use QnA answer as base, but enhance it with business context
                    qna_response = qna_match['answer']
                    
                    # Get system prompt if available
                    system_prompt = ""
                    if business_settings and business_settings.get('system_prompt'):
                        system_prompt = business_settings['system_prompt']
                    
                    # Create enhanced response using both QnA and system context
                    enhanced_system_message = f"""You are responding to a user query using knowledge from a Q&A database. 

Based on the user's question: "{chat_request.query}"
I found this relevant Q&A pair:

Question: {qna_match['question']}
Answer: {qna_response}

"""
                    
                    if system_prompt:
                        enhanced_system_message += f"Additional business context: {system_prompt}\n\n"
                    
                    if business_settings:
                        business_info = prepare_business_info(business_settings)
                        enhanced_system_message += f"Business Information:\n{business_info}\n\n"
                    
                    enhanced_system_message += """INSTRUCTIONS:
1. Use the Q&A answer as your primary response
2. Enhance it with business context if relevant
3. Keep the response natural and conversational
4. If the Q&A answer fully addresses the question, you can use it directly
5. If additional context would be helpful, integrate it smoothly
6. DO NOT mention that you're using a Q&A database
7. Make the response feel natural and personalized"""

                    # Check for greeting
                    has_greeting = check_for_greeting(chat_request.query)
                    if has_greeting:
                        enhanced_system_message += "\n\nIMPORTANT: Begin your response with a friendly greeting like 'Hello' or 'Hi there' and include an emoji right after the greeting word (with no space). For example: 'Hello👋' or 'Hi there😊'. The rest of your response should NOT contain any emojis."
                    else:
                        enhanced_system_message += "\n\nIMPORTANT: DO NOT begin your message with greeting phrases like 'hello there', 'hi there', etc. DO NOT use emojis in your response."

                    # Create messages for QnA-enhanced response
                    messages = [
                        {"role": "system", "content": enhanced_system_message},
                        {"role": "user", "content": chat_request.query}
                    ]
                    
                    # Check token limits
                    MAX_TOKENS = 16000
                    if count_tokens(messages) > MAX_TOKENS:
                        # Truncate system message if needed
                        enhanced_system_message = truncate_system_message(enhanced_system_message, 4000)
                        messages[0]["content"] = enhanced_system_message
                    
                    # Make API call for enhanced QnA response
                    try:
                        chat_completion = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages
                        )
                        
                        qna_enhanced_response = chat_completion.choices[0].message.content
                        
                        # Save to database with QnA metadata
                        await save_chat_to_db(
                            session_id=chat_request.session_id,
                            query=chat_request.query,
                            response=qna_enhanced_response,
                            is_scheduling=False,
                            qna_match_id=qna_match.get('id'),
                            qna_match_score=qna_match['score']
                        )
                        
                        print(f"Responded using QnA knowledge (ID: {qna_match.get('id')})")
                        return ChatResponse(user_id=user_id, response=qna_enhanced_response)
                        
                    except Exception as qna_error:
                        print(f"Error generating QnA-enhanced response: {qna_error}")
                        # Fall through to regular chat flow
                else:
                    print("No relevant QnA match found, proceeding with regular chat")
            else:
                print("No QnA items found for user")
        else:
            print("Scheduling intent detected, skipping QnA check")
        
        # ===== ORIGINAL CHAT LOGIC CONTINUES =====
        
        # Filter for active services
        active_services = []
        if business_settings and business_settings.get('services'):
            active_services = [s for s in business_settings['services'] if s.get('isActive', True)]
            print(f"Found {len(active_services)} active services out of {len(business_settings.get('services', []))} total services")
        
        # Prepare system message - prioritize business settings
        if business_settings:
            # Add debugging info about business settings
            print(f"Found business settings: {business_settings.get('name', 'Unnamed Business')}")
            
            # Use the token-aware function to prepare business info
            system_message = prepare_business_info(business_settings)
            
            # Enhanced system prompt for better appointment handling
            system_message += "\n\nIMPORTANT: You are a virtual assistant that helps with booking appointments. If a user wants to schedule an appointment, help them by collecting the necessary information and use the provided function to create a calendar event. ONLY discuss and recommend ACTIVE services listed above."
        else:
            # Default system message when no business settings exist
            system_message = """You are a helpful assistant that manages appointments. 
For appointment scheduling, please collect:
Can you provide the following details?
1. Meeting topic
2. Start date and time
3. End date and time
4. email address

if user talking in hebrew ask him: 

האם תוכל לספק את הפרטים הבאים?
1. נושא הפגישה
2. תאריך ושעת התחלה
3. תאריך ושעה סיום
4. כתובת אימייל

IMPORTANT: You CAN and SHOULD book appointments when requested. Use the appointment scheduling function when appropriate.
Respond in a professional tone."""

        # Set max tokens limit with buffer
        MAX_TOKENS = 16000  # Lower than the actual limit of 16385
        
        # DEBUG: Log system message beginning
        print(f"System message starts with: {system_message[:100]}...")
        
        # DEBUG: Log scheduling intent
        print(f"Scheduling intent detected: {is_scheduling}")
        
        if is_scheduling:
            # DEBUG: Log entering scheduling flow
            print("Entering scheduling flow...")
            
            # Enhanced system message specific for scheduling
            scheduling_system_message = system_message
            
            # Add active services info explicitly for scheduling
            if active_services:
                active_services_info = "\n".join(
                    f"- {s['serviceName']}: {s.get('description', '')}" 
                    for s in active_services[:9]  # Limit to 9 services
                )
                scheduling_system_message += f"\n\nONLY offer these active services for booking:\n{active_services_info}"
            else:
                scheduling_system_message += "\n\nThere are no specific services defined. Schedule a general appointment."
            
            scheduling_system_message += "\n\nThe user is trying to schedule an appointment. Help them by collecting all necessary information and use the create_calendar_event function to book it. DO NOT refuse to book appointments - that is your primary function. ONLY discuss and recommend ACTIVE services."
            
            # Define the function call with calendar event schema
            calendar_function = {
                "type": "function",
                "function": {
                    "name": "create_calendar_event",
                    "description": "Schedule a calendar appointment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "The topic of the meeting"},
                            "start_datetime": {"type": "string", "description": "Start time in ISO format (YYYY-MM-DDTHH:MM:SS)"},
                            "end_datetime": {"type": "string", "description": "End time in ISO format (YYYY-MM-DDTHH:MM:SS)"},
                            "description": {"type": "string", "description": "Additional details about the appointment"},
                            "attendees": {
                                "type": "array",
                                "items": {"type": "string", "format": "email"},
                                "description": "List of email addresses for attendees"
                            }
                        },
                        "required": ["summary", "start_datetime", "end_datetime"]
                    }
                }
            }
            
            # Create messages array
            messages = [
                {"role": "system", "content": scheduling_system_message},
                {"role": "user", "content": chat_request.query}
            ]
            
            # Check if token count exceeds limit
            message_tokens = count_tokens(messages)
            if message_tokens > MAX_TOKENS:
                # Calculate how much we need to reduce
                excess_tokens = message_tokens - MAX_TOKENS + 500  # Buffer
                
                # Recalculate with a shorter system message
                max_system_tokens = len(encoder.encode(scheduling_system_message)) - excess_tokens
                if max_system_tokens < 500:  # Minimum viable system message
                    max_system_tokens = 500
                
                scheduling_system_message = truncate_system_message(scheduling_system_message, max_system_tokens)
                messages[0]["content"] = scheduling_system_message
                
                # Verify we're now within limits
                if count_tokens(messages) > MAX_TOKENS:
                    # As a last resort, truncate the user query
                    user_tokens = len(encoder.encode(chat_request.query))
                    if user_tokens > 1000:  # Only truncate if it's long
                        max_query_tokens = user_tokens - (count_tokens(messages) - MAX_TOKENS) - 100
                        truncated_query = encoder.decode(encoder.encode(chat_request.query)[:max_query_tokens])
                        truncated_query += " [message truncated]"
                        messages[1]["content"] = truncated_query
            
            # DEBUG: Log messages before API call
            print(f"Making scheduling API call with message tokens: {count_tokens(messages)}")
            
            # Make the API call with token-managed messages
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                tools=[calendar_function],
                tool_choice="auto"  # Explicitly tell the model to use tools when appropriate
            )

            # DEBUG: Log basic response info
            print(f"Received response. Contains tool calls: {hasattr(response.choices[0].message, 'tool_calls') and bool(response.choices[0].message.tool_calls)}")
            
            message = response.choices[0].message
            
            # Check for tool calls from the API response
            if message.tool_calls:
                try:
                    # DEBUG: Log tool call details
                    print(f"Tool call received: {message.tool_calls[0].function.name}")
                    
                    event_data = json.loads(message.tool_calls[0].function.arguments)
                    print(f"Event data: {event_data}")
                    
                    # Validate that the requested service is active (if service is specified)
                    is_valid_service = True
                    if event_data.get('description') and business_settings and business_settings.get('services'):
                        # Basic check if the description contains an inactive service name
                        inactive_service_names = [s['serviceName'].lower() for s in business_settings.get('services', []) 
                                            if not s.get('isActive', True)]
                        
                        for inactive_service in inactive_service_names:
                            if inactive_service in event_data.get('description', '').lower():
                                is_valid_service = False
                                break
                    
                    if not is_valid_service:
                        # If inactive service detected, return a helpful message
                        inactive_response = "I notice you're interested in a service that isn't currently available. Here are the services we currently offer:\n\n"
                        inactive_response += "\n".join(
                            f"- {s['serviceName']}: {s.get('description', '')}" 
                            for s in active_services
                        )
                        inactive_response += "\n\nWould you like to book an appointment for one of these services instead?"
                        
                        await save_chat_to_db(
                            session_id=chat_request.session_id,
                            query=chat_request.query,
                            response=inactive_response,
                            is_scheduling=True
                        )
                        
                        return ChatResponse(
                            user_id=user_id,
                            response=inactive_response
                        )
                    
                    event_data['summary'] = event_data.get('summary') or "Appointment"
                    
                    # Make sure we have both attendees and user email if available
                    if 'attendees' not in event_data:
                        event_data['attendees'] = []
                    
                    if user.get('email') and user['email'] not in event_data['attendees']:
                        event_data['attendees'].append(user['email'])
                    
                    # Create the calendar event
                    event_result = create_calendar_event(EventRequest(**event_data))
                    
                    if event_result["status"] == "success":
                        # Format a friendly response for successful scheduling
                        start_datetime = event_data['start_datetime'].replace('T', ' at ').split('+')[0]
                        
                        response_text = f"Great! I've scheduled your appointment:\n\n"
                        response_text += f"📅 {event_data['summary']}\n"
                        response_text += f"🕒 {start_datetime}\n"
                        
                        if event_data.get('description'):
                            response_text += f"📝 {event_data['description']}\n"
                        
                        if event_result.get('meet_link'):
                            response_text += f"\n🔗 Video call link: {event_result['meet_link']}"
                        
                        # Add confirmation number
                        response_text += f"\n\nConfirmation #: {event_result['event_id'][-6:]}"
                        
                        # Save to database before returning
                        await save_chat_to_db(
                            session_id=chat_request.session_id,
                            query=chat_request.query,
                            response=response_text,
                            is_scheduling=True,
                            event_details=event_result
                        )
                        
                        return ChatResponse(
                            user_id=user_id,
                            response=response_text,
                            event_details=event_result
                        )
                    else:
                        error_response = f"I tried to schedule your appointment, but encountered an error: {event_result.get('message')}"
                        await save_chat_to_db(
                            session_id=chat_request.session_id,
                            query=chat_request.query,
                            response=error_response,
                            is_scheduling=True
                        )
                        return ChatResponse(
                            user_id=user_id,
                            response=error_response
                        )
                except Exception as tool_error:
                    # Log the specific error with the tool call
                    print(f"Error processing tool call: {str(tool_error)}")
                    traceback.print_exc()
                    
                    # Fall back to regular response
                    fallback_response = f"I encountered an issue while trying to schedule your appointment: {str(tool_error)}. Please try again with complete details including date, time, and purpose."
                    
                    await save_chat_to_db(
                        session_id=chat_request.session_id,
                        query=chat_request.query,
                        response=fallback_response,
                        is_scheduling=True
                    )
                    
                    return ChatResponse(
                        user_id=user_id,
                        response=fallback_response
                    )
            else:
                # The model didn't use the function, but we know it's a scheduling intent
                # Let's get more information and proceed with normal chat flow
                print("Model didn't use tool call despite scheduling intent. Using regular chat response.")
                
                # If the API didn't use the tool, we'll use a modified system message
                # that encourages getting the details for next time
                assist_system_message = system_message + "\n\nThe user wants to schedule an appointment. If you don't have enough details yet, ask Can I help you with a specific topic? like date, time, and purpose. DO NOT refuse to help with scheduling - that is your primary purpose. Never say you cannot book appointments. ONLY mention active services."
                
                if active_services:
                    active_services_info = "\n".join(
                        f"- {s['serviceName']}: {s.get('description', '')}" 
                        for s in active_services[:5]
                    )
                    assist_system_message += f"\n\nACTIVE SERVICES:\n{active_services_info}"
                
                messages = [
                    {"role": "system", "content": assist_system_message},
                    {"role": "user", "content": chat_request.query}
                ]
                
                # Check token count for regular messages
                if count_tokens(messages) > MAX_TOKENS:
                    # Apply token reduction as before
                    assist_system_message = truncate_system_message(assist_system_message, 4000)
                    messages[0]["content"] = assist_system_message
                
                # Use regular chat completion to ask for more details
                chat_completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                
                # Get text response
                chat_response = chat_completion.choices[0].message.content
                
                # Save to database
                await save_chat_to_db(
                    session_id=chat_request.session_id,
                    query=chat_request.query,
                    response=chat_response,
                    is_scheduling=True
                )
                
                return ChatResponse(user_id=user_id, response=chat_response)
        
        # Regular chat flow (non-scheduling intent)
        print("Using regular chat flow...")
        
        # Check if we should add greeting formatting
        has_greeting = check_for_greeting(chat_request.query)
        
        # Add greeting instructions if needed
        if has_greeting:
            system_message += "\n\nIMPORTANT: Begin your response with a friendly greeting like 'Hello' or 'Hi there' and include an emoji right after the greeting word (with no space). For example: 'Hello👋' or 'Hi there😊'. The rest of your response should NOT contain any emojis."
        else:
            system_message += "\n\nIMPORTANT: DO NOT begin your message with greeting phrases like 'hello there', 'hi there', etc. DO NOT use emojis in your response."
        
        # Create messages for regular chat flow
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": chat_request.query}
        ]
        
        # Check token count for regular messages
        message_tokens = count_tokens(messages)
        if message_tokens > MAX_TOKENS:
            # Apply same token reduction logic as before
            excess_tokens = message_tokens - MAX_TOKENS + 500  # Buffer
            max_system_tokens = len(encoder.encode(system_message)) - excess_tokens
            if max_system_tokens < 500:
                max_system_tokens = 500
            
            system_message = truncate_system_message(system_message, max_system_tokens)
            messages[0]["content"] = system_message
            
            # If still too large, truncate user query as last resort
            if count_tokens(messages) > MAX_TOKENS:
                user_tokens = len(encoder.encode(chat_request.query))
                if user_tokens > 1000:
                    max_query_tokens = user_tokens - (count_tokens(messages) - MAX_TOKENS) - 100
                    truncated_query = encoder.decode(encoder.encode(chat_request.query)[:max_query_tokens])
                    truncated_query += " [message truncated]"
                    messages[1]["content"] = truncated_query
        
        # Make regular chat API call
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Get the response content
        chat_response = chat_completion.choices[0].message.content
        
        # Save to database
        await save_chat_to_db(
            session_id=chat_request.session_id,
            query=chat_request.query,
            response=chat_response,
            is_scheduling=is_scheduling
        )

        return ChatResponse(user_id=user_id, response=chat_response)

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))











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
