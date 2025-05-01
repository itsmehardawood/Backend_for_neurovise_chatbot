from fastapi import APIRouter, Request, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from pymongo import MongoClient
from utils.utils import get_chat_completion, detect_scheduling_intent, create_calendar_event, EventRequest
from bson import ObjectId
import os
import datetime
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

twilio_router = APIRouter()

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client.Echo_db

users_collection = db.users
business_settings_collection = db.services
chat_history_collection = db.chat_history



@twilio_router.post("/twilio/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...),
    To: str = Form(...)
):
    customer_number = From.replace("whatsapp:", "").replace("+", "").strip()
    owner_number = To.replace("whatsapp:", "").replace("+", "").strip()

    print(f"[DEBUG] Customer: {customer_number}, Owner: {owner_number}")

    # Find the owner user
    owner_user = users_collection.find_one({"phone_number": owner_number})
    if not owner_user:
        print(f"[DEBUG] No matching owner user found.")
        return Response(status_code=200)

    owner_user_id = str(owner_user["_id"])

    # Get business settings
    business_settings = business_settings_collection.find_one({"user_id": ObjectId(owner_user_id)})
    
    # Prepare system message - prioritize business settings
    if business_settings:
        # Start with the custom system prompt if available
        system_message = business_settings.get('system_prompt', '')
        
        # Add services information if services exist
        if business_settings.get('services'):
            services_info = "\n".join(
                f"Service: {s['serviceName']}\nDescription: {s['description']}\nPrice: {s.get('price', 'N/A')}"
                for s in business_settings['services']
            )
            system_message = f"Available Services:\n{services_info}\n\n{system_message}"
        
        # Add tone instruction
        tone = business_settings.get('chat_tone', 'professional')
        system_message = f"{system_message}\n\nRespond in a {tone} tone."
    else:
        # Default system message when no business settings exist
        system_message = """You are Alex, a friendly assistant who speaks in a natural, conversational way. 
Use a warm, engaging tone with occasional contractions and natural language patterns like a real person would.
Avoid robotic responses and formal language. Be helpful, empathetic, and sound like you're chatting with a friend. For appointment scheduling, please collect:
1. Preferred date
2. Preferred time
3. Contact email
4. Service interested in
5. Any special requests

Respond in a professional tone."""

    try:
        # Step 1: Detect scheduling intent
        is_scheduling = await detect_scheduling_intent(Body)

        if is_scheduling:
            print("[DEBUG] Scheduling intent detected.")

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": Body}
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
                try:
                    event_data = json.loads(message.tool_calls[0].function.arguments)

                    event_data['summary'] = event_data.get('summary') or "Appointment"

                    if owner_user.get('email'):
                        event_data.setdefault('attendees', []).append(owner_user['email'])

                    event_result = create_calendar_event(EventRequest(**event_data))

                    if event_result["status"] == "success":
                        reply_text = (
                            f"✅ Scheduled: {event_data['summary']}\n"
                            f"📅 Date: {event_data['start_datetime']}\n"
                        )
                        if event_result.get('meet_link'):
                            reply_text += f"\n🔗 Meet link: {event_result['meet_link']}"
                    else:
                        reply_text = f"❌ Failed to schedule: {event_result.get('message')}"
                except Exception as e:
                    print(f"[ERROR] Failed parsing event details: {e}")
                    reply_text = "I need a bit more information to schedule. Please tell me the date, time, and your email."
            else:
                # Not enough info detected by OpenAI
                reply_text = "I need a few more details to schedule your appointment. Can you share the date, time, and your email?"

        else:
            print("[DEBUG] Regular chat.")
            # Step 2: Regular Chat (not scheduling)
            reply_text = get_chat_completion(Body, system_message)

        # Step 3: Save chat history
        chat_history_collection.update_one(
            {"customer_number": customer_number, "owner_number": owner_number},
            {
                "$setOnInsert": {
                    "customer_number": customer_number,
                    "owner_number": owner_number,
                    "name": "WhatsApp user",
                    "owner_user_id": owner_user_id
                },
                "$push": {
                    "messages": {
                        "timestamp": datetime.datetime.utcnow(),
                        "query": Body,
                        "response": reply_text
                    }
                }
            },
            upsert=True
        )

        # Step 4: Respond back
        twilio_response = MessagingResponse()
        twilio_response.message(reply_text)
        return Response(content=str(twilio_response), media_type="application/xml")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        twilio_response = MessagingResponse()
        twilio_response.message("Sorry, something went wrong.")
        return Response(content=str(twilio_response), media_type="application/xml")