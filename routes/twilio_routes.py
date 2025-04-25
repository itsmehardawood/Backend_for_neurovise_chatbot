from fastapi import APIRouter, Request, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from utils.utils import get_chat_completion
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from src.constant import system_message
import datetime

load_dotenv()

twilio_router = APIRouter()

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["Echo_db"]
users_collection = db["users"]
chat_collection = db["whatsapp_chat"]


@twilio_router.post("/twilio/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...)
):
    # Normalize phone number (remove "whatsapp:" and "+")
    phone_number = From.replace("whatsapp:", "").replace("+", "").strip()
    print(f"[DEBUG] Looking for number: {repr(phone_number)}")

    # Find user
    user = users_collection.find_one({"phone_number": phone_number})

    if user:
        user_id = str(user["_id"])
        email = user.get("email", "unknown")
        name = "WhatsApp user"

        # Generate chatbot reply
        reply = get_chat_completion(Body, system_message)

        # Message structure
        message_entry = {
            "timestamp": datetime.datetime.utcnow(),
            "user_message": Body,
            "assistant_reply": reply
        }

        # Upsert conversation into chat history array
        chat_collection.update_one(
            {"user_id": user_id},
            {
                "$setOnInsert": {
                    "user_id": user_id,
                    "name": name,
                    "email": email,
                    "phone_number": phone_number
                },
                "$push": {
                    "messages": message_entry
                }
            },
            upsert=True
        )

        # Twilio response
        twilio_response = MessagingResponse()
        twilio_response.message(reply)
        return Response(content=str(twilio_response), media_type="application/xml")
    else:
        print(f"[DEBUG] No matching user found for phone number: {phone_number}")
        return Response(status_code=200)
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