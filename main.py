from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.schema import ChatRequest, ChatResponse
from utils.utils import get_chat_completion, check_for_greeting, check_for_proper_greeting, remove_greeting
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
from utils.utils import get_chat_completion
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

db = client.Echo_db
users_collection = db.users
business_settings_collection = db.services
chat_history_collection = db.chat_history
chat_sessions_collection = db.chat_sessions


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
                "price": f"{service.price:.2f}",  # Properly formatted as string
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
        price = service["price"]
        if isinstance(price, str):
            try:
                price = Decimal(price)
            except:
                price = Decimal("0.00")
        
        # Return properly formatted service data
        return {
            "serviceName": service["serviceName"],
            "description": service["description"],
            "priceType": service["priceType"],
            "price": price,
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
async def update_service(service_id: str, service_update: ServiceUpdate = Body(...)):
    try:
        # Attempt to cast service_id to int for matching
        try:
            service_id_int = int(service_id)
            query = {"services.id": service_id_int}
        except ValueError:
            query = {"services.id": service_id}

        # Retrieve the document containing the service
        business_settings = await business_settings_collection.find_one(query)
        if not business_settings:
            raise HTTPException(status_code=404, detail="Service not found")

        # Prepare the update operations
        update_data = service_update.dict(exclude_unset=True)
        update_operations = {}

        for field, value in update_data.items():
            if field == "working_hours":
                for day, hours in value.items():
                    if hasattr(hours, "dict"):
                        hours_dict = hours.dict(exclude_unset=True)
                    else:
                        hours_dict = hours

                    for hour_field, hour_value in hours_dict.items():
                        if hour_value is not None:
                            update_operations[f"services.$.working_hours.{day}.{hour_field}"] = hour_value
            else:
                if value is not None:
                    if field == "price" and isinstance(value, Decimal):
                        value = float(value)
                    update_operations[f"services.$.{field}"] = value

        if not update_operations:
            raise HTTPException(status_code=400, detail="No valid fields to update")

        # Perform the update in the array of embedded documents
        result = await business_settings_collection.update_one(
            query,
            {"$set": update_operations}
        )

        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Service not found or no changes made")

        # Fetch the updated document to return the modified service
        updated_doc = await business_settings_collection.find_one(query)
        updated_service = None
        for service in updated_doc.get("services", []):
            if str(service.get("id")) == str(service_id):
                updated_service = service
                break

        if not updated_service:
            raise HTTPException(status_code=404, detail="Updated service not found")

        # Convert Decimal to float for JSON serialization
        if isinstance(updated_service.get("price"), Decimal):
            updated_service["price"] = float(updated_service["price"])

        return updated_service

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



# delete api for service 


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
    
    
    
    
    
    
# old chat


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
        
#         # Prepare system message - prioritize business settings
#         if business_settings:
#             # Start with the custom system prompt if available
#             system_message = business_settings.get('system_prompt', '')
            
#             # Add services information if services exist
#             if business_settings.get('services'):
#                 services_info = "\n".join(
#                     f"Service: {s['serviceName']}\nDescription: {s['description']}\nPrice: {s.get('price', 'N/A')}"
#                     for s in business_settings['services']
#                 )
#                 system_message = f"Available Services:\n{services_info}\n\n{system_message}"
            
#             # Add tone instruction
#             tone = business_settings.get('chat_tone', 'professional')
#             system_message = f"{system_message}\n\nRespond in a {tone} tone."
#         else:
#             # Default system message when no business settings exist
#             system_message = """You are a helpful assistant. For appointment scheduling, please collect:
# 1. Preferred date
# 2. Preferred time
# 3. Contact email
# 4. Service interested in
# 5. Any special requests

# Respond in a professional tone."""

#         # [Rest of your existing code remains the same...]
#         # Determine intent and process
#         is_scheduling = await detect_scheduling_intent(chat_request.query)
        
#         if is_scheduling:
#             # Scheduling flow
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": system_message},
#                     {"role": "user", "content": chat_request.query}
#                 ],
#                 tools=[{
#                     "type": "function",
#                     "function": {
#                         "name": "create_calendar_event",
#                         "description": "Schedule a calendar appointment",
#                         "parameters": {
#                             "type": "object",
#                             "properties": {
#                                 "summary": {"type": "string"},
#                                 "start_datetime": {"type": "string"},
#                                 "end_datetime": {"type": "string"},
#                                 "description": {"type": "string"},
#                                 "attendees": {
#                                     "type": "array",
#                                     "items": {"type": "string", "format": "email"}
#                                 }
#                             },
#                             "required": ["summary", "start_datetime", "end_datetime"]
#                         }
#                     }
#                 }]
#             )

#             message = response.choices[0].message
#             if message.tool_calls:
#                 event_data = json.loads(message.tool_calls[0].function.arguments)
#                 event_data['summary'] = event_data.get('summary') or "Appointment"
#                 if user.get('email'):
#                     event_data.setdefault('attendees', []).append(user['email'])
                
#                 event_result = create_calendar_event(EventRequest(**event_data))
                
#                 if event_result["status"] == "success":
#                     response_text = (f"Scheduled: {event_data['summary']}\n"
#                                     f"Date: {event_data['start_datetime']}\n")
#                     if event_result.get('meet_link'):
#                         response_text += f"\nMeet link: {event_result['meet_link']}"
                    
#                     # Save to database before returning
#                     await save_chat_to_db(
#                         session_id=chat_request.session_id,
#                         query=chat_request.query,
#                         response=response_text,
#                         is_scheduling=True,
#                         event_details=event_result
#                     )
                    
#                     return ChatResponse(
#                         user_id=user_id,
#                         response=response_text,
#                         event_details=event_result
#                     )
#                 else:
#                     error_response = f"Failed to schedule: {event_result.get('message')}"
#                     await save_chat_to_db(
#                         session_id=chat_request.session_id,
#                         query=chat_request.query,
#                         response=error_response,
#                         is_scheduling=True
#                     )
#                     return ChatResponse(
#                         user_id=user_id,
#                         response=error_response
#                     )
        
#         # Regular chat flow
#         chat_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": chat_request.query}
#             ]
#         ).choices[0].message.content

#         # Save to database
#         await save_chat_to_db(
#             session_id=chat_request.session_id,
#             query=chat_request.query,
#             response=chat_response,
#             is_scheduling=is_scheduling
#         )

#         return ChatResponse(user_id=user_id, response=chat_response)

#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))


    
    
 




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

async def save_chat_to_db(session_id: str, query: str, response: str, is_scheduling: bool, event_details: Optional[dict] = None):
    """Helper function to save chat history to database"""
    try:
        update_data = {
            "$push": {
                "messages": {
                    "query": query,
                    "response": response,
                    "timestamp": datetime.utcnow(),
                    "is_scheduling": is_scheduling
                }
            },
            "$set": {"last_activity": datetime.utcnow()}
        }
        
        if event_details:
            update_data["$push"]["messages"]["event_details"] = event_details
        
        await chat_sessions_collection.update_one(
            {"session_id": session_id},
            update_data
        )
    except Exception as e:
        print(f"Failed to save chat to database: {str(e)}")
        traceback.print_exc()




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
        max_services = 5  # Adjust based on your needs
        services = active_services[:max_services]
        
        if services:  # Only proceed if there are active services
            services_info = "\n".join(
                f"- {s['serviceName']}: {s.get('price', 'N/A')} - {s.get('description', '')[:100]}..." 
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
1. Preferred date
2. Preferred time
3. Contact email
4. Service interested in
5. Any special requests

IMPORTANT: You CAN and SHOULD book appointments when requested. Use the appointment scheduling function when appropriate.
Respond in a professional tone."""

        # Set max tokens limit with buffer
        MAX_TOKENS = 16000  # Lower than the actual limit of 16385
        
        # DEBUG: Log system message beginning
        print(f"System message starts with: {system_message[:100]}...")
        
        # Determine intent and process
        is_scheduling = await detect_scheduling_intent(chat_request.query)
        
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
                    f"- {s['serviceName']}: {s.get('price', 'N/A')}" 
                    for s in active_services[:5]  # Limit to 5 services
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
                            "summary": {"type": "string", "description": "Title of the appointment"},
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
                            f"- {s['serviceName']}: {s.get('price', 'N/A')}" 
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
                assist_system_message = system_message + "\n\nThe user wants to schedule an appointment. If you don't have enough details yet, ask for specific information like date, time, and purpose. DO NOT refuse to help with scheduling - that is your primary purpose. Never say you cannot book appointments. ONLY mention active services."
                
                if active_services:
                    active_services_info = "\n".join(
                        f"- {s['serviceName']}: {s.get('price', 'N/A')}" 
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
                
                # Apply greeting formatting using the functions from paste-2.txt
        
                
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
