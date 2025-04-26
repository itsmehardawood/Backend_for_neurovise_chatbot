from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.schema import ChatRequest, ChatResponse
from utils.utils import get_chat_completion
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

load_dotenv()
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
# DB_NAME=os.getenv("DB_NAME")
# users_collection= os.getenv("USERS_COLLECTION")
# services_collection= os.getenv("SERVICES_COLLECTION")

# # MongoDB(MONGO_URI)
# client = AsyncIOMotorClient(MONGO_URI)
# db = client.Echo_db
# users_collection = db.users
# business_settings_collection = db.service
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

        # Update document with combined services
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
        business_settings = await business_settings_collection.find_one(query)
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
            }
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
    
    
    
    
    
    
    
    
    
    
class ChatRequest(BaseModel):
    session_id: str
    query: str
    user_id: Optional[str] = None  # Optional for backward compatibility








from fastapi import HTTPException
from bson import ObjectId
from datetime import datetime





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

        # Prepare system message with business info
        business_settings = await business_settings_collection.find_one({"user_id": ObjectId(user_id)})
        services_info = "\n".join(
            f"Service: {s['serviceName']}\nDescription: {s['description']}\nPrice: {s.get('price', 'N/A')}"
            for s in business_settings.get("services", [])
        )
        
        system_message = f"""Business services:\n{services_info}
Respond in {business_settings.get('chat_tone', 'professional')} tone.
For appointments, collect:
1. Date
2. Start/end times
3. Email address
4. Description (optional)"""

        # Determine intent and process
        is_scheduling = await detect_scheduling_intent(chat_request.query)
        
        if is_scheduling:
            # Scheduling flow
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": chat_request.query}
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "create_calendar_event",
                        "description": "Schedule a calendar appointment",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "summary": {"type": "string"},
                                "start_datetime": {"type": "string"},
                                "end_datetime": {"type": "string"},
                                "description": {"type": "string"},
                                "attendees": {
                                    "type": "array",
                                    "items": {"type": "string", "format": "email"}
                                }
                            },
                            "required": ["summary", "start_datetime", "end_datetime"]
                        }
                    }
                }]
            )

            message = response.choices[0].message
            if message.tool_calls:
                event_data = json.loads(message.tool_calls[0].function.arguments)
                event_data['summary'] = event_data.get('summary') or "Appointment"
                if user.get('email'):
                    event_data.setdefault('attendees', []).append(user['email'])
                
                event_result = create_calendar_event(EventRequest(**event_data))
                
                if event_result["status"] == "success":
                    response_text = (f"Scheduled: {event_data['summary']}\n"
                                    f"Date: {event_data['start_datetime']}\n")
                    if event_result.get('meet_link'):
                        response_text += f"\nMeet link: {event_result['meet_link']}"
                    
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
                    error_response = f"Failed to schedule: {event_result.get('message')}"
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
        
        # Regular chat flow
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": chat_request.query}
            ]
        ).choices[0].message.content

        # Save to database
        await save_chat_to_db(
            session_id=chat_request.session_id,
            query=chat_request.query,
            response=chat_response,
            is_scheduling=is_scheduling
        )

        return ChatResponse(user_id=user_id, response=chat_response)

    except Exception as e:
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
            "customer_number": whatsapp_chat.get("customer_number", ""),
            "owner_number": whatsapp_chat.get("owner_number", ""),
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


# @app.get("/chat-sessions/{user_id}")
# async def get_chat_sessions(user_id: str):
#     # 1. Fetch regular chat sessions
#     sessions_cursor = chat_sessions_collection.find({"user_id": user_id})
#     sessions = []
#     async for session in sessions_cursor:
#         session["_id"] = str(session["_id"])
#         sessions.append(session)
#     # 2. Fetch WhatsApp chat history
#     whatsapp_chat = await chat_history_collection.find_one({"user_id": user_id})
#     whatsapp_session = None
#     if whatsapp_chat and whatsapp_chat.get("messages"):
#         whatsapp_session = {
#             "_id": str(whatsapp_chat["_id"]),
#             "session_id": "whatsapp_chat",
#             "full_name": whatsapp_chat.get("name", "WhatsApp User"),
#             "email": whatsapp_chat.get("email", ""),
#             "phone_number": whatsapp_chat.get("phone_number", ""),
#             "created_at": whatsapp_chat["messages"][0]["timestamp"] if whatsapp_chat["messages"] else None,
#             "messages": [
#                 {
#                     "query": msg.get("user_message", ""),
#                     "response": msg.get("assistant_reply", ""),
#                     "timestamp": msg.get("timestamp", None)
#                 }
#                 for msg in whatsapp_chat["messages"]
#             ]
#         }
#         sessions.append(whatsapp_session)

#     # 3. Raise 404 only if both are missing
#     if not sessions:
#         raise HTTPException(status_code=404, detail="No chat sessions or WhatsApp chat history found for this user.")

#     # 4. Return unified session list
#     return {
#         "user_id": user_id,
#         "chat_sessions": sessions
#     }


# @app.get("/chat-sessions/{user_id}")
# async def get_chat_sessions(user_id: str):
#     sessions_cursor = chat_sessions_collection.find({"user_id": user_id})
#     sessions = []
#     async for session in sessions_cursor:
#         session["_id"] = str(session["_id"])  # convert ObjectId to string
#         sessions.append(session)

#     if not sessions:
#         raise HTTPException(status_code=404, detail="No chat sessions found for this user.")

#     return {"user_id": user_id, "chat_sessions": sessions}
