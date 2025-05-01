from fastapi import APIRouter, Request, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from pymongo import MongoClient
from utils.utils import get_chat_completion, detect_scheduling_intent, create_calendar_event, EventRequest
from bson import ObjectId
import os
import datetime
import json
import re
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


def is_service_inquiry(text):
    service_keywords = [
        "services", "service", "offer", "providing", "do you have", 
        "what do you do", "what do you provide", "options", 
        "what are your", "tell me about your", "tell me what"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in service_keywords)


def is_greeting(text):
    greetings = ["hi", "hello", "hey", "yo", "good morning", "good evening", "good afternoon"]
    text = text.lower().strip()
    return any(re.fullmatch(rf"{greet}", text) for greet in greetings)


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

    owner_user = users_collection.find_one({"phone_number": owner_number})
    if not owner_user:
        print(f"[DEBUG] No matching owner user found.")
        return Response(status_code=200)

    owner_user_id = str(owner_user["_id"])

    business_settings = business_settings_collection.find_one({"user_id": ObjectId(owner_user_id)})
    
    base_human_message = """You are a friendly person having a casual conversation. 

IMPORTANT: You must sound like a real human, not a chatbot or AI assistant. 

AVOID completely:
- "How can I assist you today?"
- "I'm here to help"
- "How may I help you?"
- "I'd be happy to assist"
- "Is there anything else you need help with?"
- Any variation of these standard assistant phrases

DO THIS INSTEAD:
- Use casual, conversational language like "Got it"
- Talk like a friend would, not a service provider
- Use contractions (I'm, you're, we'll, that's)
- Be naturally helpful without saying you're "here to help"
- Be concise - humans don't write long, perfect paragraphs
- Occasionally use sentence fragments, like real texting
- Remember that less formal = more human sounding

Keep responses brief like a text message - not essay-length."""

    services_section = ""
    has_services = False

    if business_settings:
        custom_prompt = business_settings.get('system_prompt', '')

        if business_settings.get('services') and isinstance(business_settings.get('services'), list):
            valid_services = []
            for service in business_settings['services']:
                if (isinstance(service, dict) and 
                    service.get('serviceName') and 
                    service.get('description') and 
                    service.get('isActive', False) == True):
                    valid_services.append(service)

            if valid_services:
                has_services = True
                services_info = "\n".join(
                    f"Service: {s['serviceName']}\nDescription: {s['description']}\nPrice: {s.get('price', 'N/A')}"
                    for s in valid_services
                )
                services_section = f"Available Services:\n{services_info}"
                print(f"[DEBUG] Found {len(valid_services)} active services")

        tone = business_settings.get('chat_tone', 'professional')

        if services_section:
            system_message = f"{base_human_message}\n\n{custom_prompt}\n\n{services_section}\n\nRespond in a {tone}, human-like tone."
        else:
            system_message = f"{base_human_message}\n\n{custom_prompt}\n\nRespond in a {tone}, human-like tone."
    else:
        system_message = f"""{base_human_message}

For appointment scheduling, please collect:
1. Preferred date
2. Preferred time
3. Contact email
4. Service interested in
5. Any special requests

Keep it conversational and friendly - like texting a colleague."""

    try:
        if is_service_inquiry(Body):
            print("[DEBUG] Service inquiry detected.")
            if has_services:
                enhanced_system_message = system_message + "\n\nThe user is asking about services. Talk about the available services in a friendly way."
                reply_text = get_chat_completion(Body, enhanced_system_message)
            else:
                no_services_message = base_human_message + "\n\nThe user is asking about services, but you don't have any active services configured yet. Politely explain that you're still setting up your services and will have more information soon. Suggest they can schedule a consultation to discuss their needs if they'd like."
                reply_text = get_chat_completion(Body, no_services_message)

        elif is_greeting(Body):
            print("[DEBUG] Greeting detected.")
            reply_text = get_chat_completion(Body, system_message)

        else:
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
                                f"✅ All set! {event_data['summary']} scheduled for {event_data['start_datetime'].split('T')[0]} at {event_data['start_datetime'].split('T')[1][:5]}"
                            )
                            if event_result.get('meet_link'):
                                reply_text += f"\n\n🔗 Here's your meeting link: {event_result['meet_link']}"
                        else:
                            reply_text = f"Hmm, couldn't schedule that. {event_result.get('message', 'Try again?')}"
                    except Exception as e:
                        print(f"[ERROR] Failed parsing event details: {e}")
                        reply_text = "Need a bit more info for scheduling. What date and time works for you? And your email?"
                else:
                    reply_text = "Just need a few more details to get you scheduled. What date/time works? And what's your email?"
            else:
                print("[DEBUG] Regular chat.")
                reply_text = get_chat_completion(Body, system_message)

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

        twilio_response = MessagingResponse()
        twilio_response.message(reply_text)
        return Response(content=str(twilio_response), media_type="application/xml")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        twilio_response = MessagingResponse()
        twilio_response.message("Oops! Something went wrong on our end. Mind trying again?")
        return Response(content=str(twilio_response), media_type="application/xml")
