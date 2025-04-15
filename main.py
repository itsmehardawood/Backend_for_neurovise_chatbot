from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.schema import ChatRequest, ChatResponse
from utils.utils import get_chat_completion
from dotenv import load_dotenv
from routes.twilio_routes import twilio_router
import os
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


from fastapi import APIRouter, HTTPException
from bson import ObjectId

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        # Convert user_id string to ObjectId
        try:
            user_id = ObjectId(chat_request.user_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid user_id format")

        # Fetch user data using the provided user_id
        user = await users_collection.find_one({"_id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch business settings for the user
        business_settings = await business_settings_collection.find_one({"user_id": user_id})
        if not business_settings:
            raise HTTPException(status_code=404, detail="Business settings not found for the user")

        # Assuming services are embedded inside the business settings, extract services
        services = business_settings.get("services", [])
        if not services:
            raise HTTPException(status_code=404, detail="Services not found for the user")

        # Customize the system message with business services, working hours, and prices
        services_info = []
        for service in services:
            service_info = f"Service: {service['serviceName']}\nDescription: {service['description']}\n"

            # Add price information
            if 'price' in service:
                service_info += f"Price: ${service['price']}\n"
            else:
                service_info += "Price: Not specified\n"

            # Add working hours if available
            if service.get('working_hours'):
                working_hours_info = ", ".join(
                    [f"{day}: {hours['start']} - {hours['end']}" for day, hours in service['working_hours'].items() if
                     hours.get('active')]
                )
                service_info += f"Working hours: {working_hours_info if working_hours_info else 'Not specified'}\n"

            services_info.append(service_info)

        services_details = "\n".join(services_info) if services_info else "No services available."

        # Get chat tone setting (customize the response based on user preferences)
        chat_tone = business_settings.get('chat_tone', 'default')  # Default tone if not found

        # Customize system message with services, prices, and tone
        system_message = f"The business offers the following services:\n{services_details}\n"
        system_message += f"Please respond in a {chat_tone} tone."

        # Get the chat response from OpenAI (or any other service you're using)
        response = get_chat_completion(chat_request.query, system_message)

        return ChatResponse(
            user_id=chat_request.user_id,
            response=response
        )
    except HTTPException as e:
        print(f"HTTP error: {e.detail}")  # Log the error message for debugging
        raise e
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Log unexpected errors
        raise HTTPException(status_code=500, detail="Internal server error")

port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)