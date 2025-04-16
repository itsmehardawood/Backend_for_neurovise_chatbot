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
MONGO_URI = os.getenv("MONGO_URI")
# DB_NAME=os.getenv("DB_NAME")
# users_collection= os.getenv("USERS_COLLECTION")
# services_collection= os.getenv("SERVICES_COLLECTION")

# # MongoDB(MONGO_URI)
# client = AsyncIOMotorClient(MONGO_URI)
# db = client.Echo_db
# users_collection = db.users
# business_settings_collection = db.services
from motor.motor_asyncio import AsyncIOMotorClient
import ssl


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
    hashed_pw = hash_password(user.password)
    await users_collection.insert_one({"email": user.email, "password": hashed_pw})
    return {"message": "User registered successfully"}

@app.post("/login", response_model=LoginResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    This endpoint is used by Swagger UI's "Authorize" dialog.
    It expects form fields: username, password, grant_type=password.
    """
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

@app.post("/logout")
async def logout():
    """
    Stateless logout: just instruct the client to delete the token.
    """
    return {"message": "Logout successful. Please remove the token on the client side."}


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
                "price": float(service.price),  # Convert Decimal to float
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



import uuid
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from bson import ObjectId
import os


from fastapi import FastAPI
from pydantic import BaseModel, EmailStr
from uuid import uuid4
from datetime import datetime

class StartChatRequest(BaseModel):
    full_name: str
    email: EmailStr
    phone_number: str

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
        "messages": []
    }

    await chat_sessions_collection.insert_one(new_chat_session)

    return StartChatResponse(
        session_id=session_id,
        message="Chat session started"
    )
class ChatRequest(BaseModel):
    session_id: str
    query: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        # Lookup session by session_id
        session = await chat_sessions_collection.find_one({"session_id": chat_request.session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        # Generate chat response
        response = get_chat_completion(chat_request.query, "You are chatting with our business assistant.")

        # Add chat entry
        chat_doc = {
            "query": chat_request.query,
            "response": response,
            "timestamp": datetime.utcnow()
        }

        await chat_sessions_collection.update_one(
            {"session_id": chat_request.session_id},
            {"$push": {"messages": chat_doc}}
        )

        return ChatResponse(
            user_id=str(session["_id"]),
            response=response
        )

    except HTTPException as e:
        print(f"HTTP error: {e.detail}")
        raise e
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(chat_request: ChatRequest):
#     try:
#         try:
#             user_id = ObjectId(chat_request.user_id)
#         except Exception:
#             raise HTTPException(status_code=400, detail="Invalid user_id format")

#         user = await users_collection.find_one({"_id": user_id})
#         if not user:
#             raise HTTPException(status_code=404, detail="User not found")

#         business_settings = await business_settings_collection.find_one({"user_id": user_id})
#         if not business_settings:
#             raise HTTPException(status_code=404, detail="Business settings not found for the user")

#         services = business_settings.get("services", [])
#         if not services:
#             raise HTTPException(status_code=404, detail="Services not found for the user")

#         services_info = []
#         for service in services:
#             service_info = f"Service: {service['serviceName']}\nDescription: {service['description']}\n"
#             if 'price' in service:
#                 service_info += f"Price: ${service['price']}\n"
#             else:
#                 service_info += "Price: Not specified\n"

#             if service.get('working_hours'):
#                 working_hours_info = ", ".join(
#                     [f"{day}: {hours['start']} - {hours['end']}" for day, hours in service['working_hours'].items() if
#                      hours.get('active')]
#                 )
#                 service_info += f"Working hours: {working_hours_info if working_hours_info else 'Not specified'}\n"

#             services_info.append(service_info)

#         services_details = "\n".join(services_info) if services_info else "No services available."
#         chat_tone = business_settings.get('chat_tone', 'default')

#         system_message = f"The business offers the following services:\n{services_details}\n"
#         system_message += f"Please respond in a {chat_tone} tone."

#         # Get chat response from model
#         response = get_chat_completion(chat_request.query, system_message)

#         current_time = datetime.utcnow()
#         one_hour_ago = current_time - timedelta(hours=1)

#         # Find latest session for the user
#         latest_session = await chat_history_collection.find_one(
#             {"user_id": user_id},
#             sort=[("last_activity", -1)]
#         )

#         chat_doc = {
#             "query": chat_request.query,
#             "response": response,
#             "timestamp": current_time
#         }

#         if latest_session and latest_session["last_activity"] > one_hour_ago:
#             # Append to the existing session
#             await chat_history_collection.update_one(
#                 {"_id": latest_session["_id"]},
#                 {
#                     "$push": {"messages": chat_doc},
#                     "$set": {"last_activity": current_time}
#                 }
#             )
#         else:
#             # Start a new session
#             new_session_id = str(uuid.uuid4())
#             await chat_history_collection.insert_one({
#                 "user_id": user_id,
#                 "session_id": new_session_id,
#                 "last_activity": current_time,
#                 "messages": [chat_doc]
#             })

#         return ChatResponse(
#             user_id=chat_request.user_id,
#             response=response
#         )

#     except HTTPException as e:
#         print(f"HTTP error: {e.detail}")
#         raise e
#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# Run with Uvicorn
port = int(os.getenv("PORT", 8000))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
