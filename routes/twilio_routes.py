from fastapi import APIRouter, Request, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from utils.utils import get_chat_completion
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime
from bson import ObjectId

load_dotenv()

twilio_router = APIRouter()

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client.Echo_db

users_collection = db.users
business_settings_collection = db.services
chat_history_collection = db.chat_history
chat_sessions_collection = db.chat_sessions

@twilio_router.post("/twilio/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...)
):
    # Normalize phone number
    phone_number = From.replace("whatsapp:", "").replace("+", "").strip()
    print(f"[DEBUG] Looking for number: {repr(phone_number)}")

    # Find the user
    user = users_collection.find_one({"phone_number": phone_number})
    if not user:
        print(f"[DEBUG] No matching user found for phone number: {phone_number}")
        return Response(status_code=200)

    user_id = str(user["_id"])
    print("user id", user_id)
    email = user.get("email", "unknown")
    name = user.get("name", "WhatsApp user")

    # Find business settings
    business_settings = business_settings_collection.find_one({"user_id": ObjectId(user_id)})
    if not business_settings:
        print(f"[DEBUG] No business settings found for user {user_id}")
        return Response(status_code=200)

    # Format services info
    services_info = "\n".join(
        f"Service: {s['serviceName']}\nDescription: {s['description']}\nPrice: {s.get('price', 'N/A')}"
        for s in business_settings.get("services", [])
    )

    # Create dynamic system message
    system_message = f"""Business services:\n{services_info}
Respond in {business_settings.get('chat_tone', 'professional')} tone.
If the user wants to book an appointment, ask for:
1. Date
2. Start/end times
3. Email address
4. Description (optional)"""

    # Generate chatbot reply
    reply = get_chat_completion(Body, system_message)

    # Store chat history
    message_entry = {
        "timestamp": datetime.datetime.utcnow(),
        "user_message": Body,
        "assistant_reply": reply
    }

    chat_history_collection.update_one(
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

    # Send Twilio WhatsApp reply
    twilio_response = MessagingResponse()
    twilio_response.message(reply)
    return Response(content=str(twilio_response), media_type="application/xml")
