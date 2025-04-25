from fastapi import APIRouter, Request, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from utils.utils import get_chat_completion
import os
from dotenv import load_dotenv
from src.constant import system_message
load_dotenv()


twilio_router = APIRouter()

# @twilio_router.post("/twilio/whatsapp")
# async def whatsapp_webhook(
#     request: Request,
#     Body: str = Form(...),
#     From: str = Form(...)
# ):
#     # Generate chatbot response
#     reply = get_chat_completion(Body, system_message)

#     # Prepare Twilio responsed
#     twilio_response = MessagingResponse()
#     twilio_response.message(reply)

#     return Response(content=str(twilio_response), media_type="application/xml")
from fastapi import APIRouter, Form, Request, Response
from twilio.twiml.messaging_response import MessagingResponse
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv("MONGO_URI")

# MongoDB setup
client = AsyncIOMotorClient(
    MONGO_URI,
    tls=True
)
db = client.Echo_db
users_collection = db.users

twilio_router = APIRouter()

@twilio_router.post("/twilio/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...)
):
    # Extract the phone number from the "From" field
    user_phone_number = From.strip()
    print("phone number of twilo get", user_phone_number)
    # Check if the phone number exists in the users collection in MongoDB
    user = await users_collection.find_one({"phone": user_phone_number})

    if user:
        # Phone number is valid, generate chatbot response
        reply = get_chat_completion(Body, system_message)

        # Prepare Twilio response
        twilio_response = MessagingResponse()
        twilio_response.message(reply)

        return Response(content=str(twilio_response), media_type="application/xml")
    else:
        # Phone number is not registered, do not respond
        return Response(content="", media_type="application/xml")
