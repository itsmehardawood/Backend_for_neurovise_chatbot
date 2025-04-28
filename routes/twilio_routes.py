# from fastapi import APIRouter, Request, Form, Response
# from twilio.twiml.messaging_response import MessagingResponse
# from utils.utils import get_chat_completion
# from pymongo import MongoClient
# import os
# from dotenv import load_dotenv
# import datetime
# from bson import ObjectId

# load_dotenv()

# twilio_router = APIRouter()

# # MongoDB connection
# mongo_uri = os.getenv("MONGO_URI")
# client = MongoClient(mongo_uri)
# db = client.Echo_db

# users_collection = db.users
# business_settings_collection = db.services
# chat_history_collection = db.chat_history

# @twilio_router.post("/twilio/whatsapp")
# async def whatsapp_webhook(
#     request: Request,
#     Body: str = Form(...),
#     From: str = Form(...),
#     To: str = Form(...)
# ):
#     # Normalize customer phone number and Twilio (owner) phone number
#     customer_number = From.replace("whatsapp:", "").replace("+", "").strip()
#     owner_number = To.replace("whatsapp:", "").replace("+", "").strip()

#     print(f"[DEBUG] Customer: {customer_number}, Owner (Twilio): {owner_number}")

#     # Find the owner user by Twilio number
#     owner_user = users_collection.find_one({"phone_number": owner_number})
#     if not owner_user:
#         print(f"[DEBUG] No matching business user found for owner number: {owner_number}")
#         return Response(status_code=200)

#     owner_user_id = str(owner_user["_id"])

#     # Find business settings by owner user_id
#     business_settings = business_settings_collection.find_one({"user_id": ObjectId(owner_user_id)})
#     if not business_settings:
#         print(f"[DEBUG] No business settings found for owner user {owner_user_id}")
#         return Response(status_code=200)

#     # Prepare services info for system prompt
#     services_info = "\n".join(
#         f"Service: {s['serviceName']}\nDescription: {s['description']}\nPrice: {s.get('price', 'N/A')}"
#         for s in business_settings.get("services", [])
#     )

#     system_message = f"""Business services:\n{services_info}
# Respond in {business_settings.get('chat_tone', 'professional')} tone.
# If the user wants to book an appointment, ask for:
# 1. Date
# 2. Start/end times
# 3. Email address
# 4. Description (optional)"""

#     # Generate chatbot reply
#     assistant_reply = get_chat_completion(Body, system_message)

#     # Prepare chat history structure
#     chat_history_entry = {
#         "customer_number": customer_number,
#         "owner_number": owner_number,
#         "name": "WhatsApp user",  # Default name
#         "owner_user_id": owner_user_id,
#         "messages": [
#             {
#                 "timestamp": datetime.datetime.utcnow(),
#                 "query": Body,
#                 "response": assistant_reply
#             }
#         ]
#     }

#     # Upsert chat history
#     chat_history_collection.update_one(
#         {
#             "customer_number": customer_number,
#             "owner_number": owner_number
#         },
#         {
#             "$setOnInsert": {
#                 "customer_number": customer_number,
#                 "owner_number": owner_number,
#                 "name": "WhatsApp user",
#                 "owner_user_id": owner_user_id
#             },
#             "$push": {
#                 "messages": {
#                     "timestamp": datetime.datetime.utcnow(),
#                     "query": Body,
#                     "response": assistant_reply
#                 }
#             }
#         },
#         upsert=True
#     )

#     # Send reply
#     twilio_response = MessagingResponse()
#     twilio_response.message(assistant_reply)
#     return Response(content=str(twilio_response), media_type="application/xml")




from fastapi import APIRouter, Request, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from utils.utils import get_chat_completion
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime
from bson import ObjectId
import json
from utils.utils import detect_scheduling_intent, create_calendar_event, EventRequest


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

    print(f"[DEBUG] Customer: {customer_number}, Owner (Twilio): {owner_number}")

    owner_user = users_collection.find_one({"phone_number": owner_number})
    if not owner_user:
        print(f"[DEBUG] No matching business user found for owner number: {owner_number}")
        return Response(status_code=200)

    owner_user_id = str(owner_user["_id"])

    business_settings = business_settings_collection.find_one({"user_id": ObjectId(owner_user_id)})
    if not business_settings:
        print(f"[DEBUG] No business settings found for owner user {owner_user_id}")
        return Response(status_code=200)

    services_info = "\n".join(
        f"Service: {s['serviceName']}\nDescription: {s['description']}\nPrice: {s.get('price', 'N/A')}"
        for s in business_settings.get("services", [])
    )

    system_message = f"""Business services:\n{services_info}
Respond in {business_settings.get('chat_tone', 'professional')} tone.
If the user wants to book an appointment, ask for:
1. Date
2. Start/end times
3. Email address
4. Description (optional)"""

    try:
        # Step 1: Detect scheduling intent
        is_scheduling = await detect_scheduling_intent(Body)

        if is_scheduling:
            # Step 2: Extract event details
            from openai import OpenAI
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
                event_data = json.loads(message.tool_calls[0].function.arguments)
                event_data['summary'] = event_data.get('summary') or "Appointment"
                if owner_user.get('email'):
                    event_data.setdefault('attendees', []).append(owner_user['email'])

                event_result = create_calendar_event(EventRequest(**event_data))

                if event_result["status"] == "success":
                    reply_text = (f"Scheduled: {event_data['summary']}\n"
                                  f"Date: {event_data['start_datetime']}\n")
                    if event_result.get('meet_link'):
                        reply_text += f"\nMeet link: {event_result['meet_link']}"
                else:
                    reply_text = f"Failed to schedule: {event_result.get('message')}"

            else:
                reply_text = "Sorry, I couldn't gather enough details to create the event."

        else:
            # Step 3: Regular Chat (not scheduling)
            reply_text = get_chat_completion(Body, system_message)

        # Step 4: Save chat history
        chat_history_collection.update_one(
            {
                "customer_number": customer_number,
                "owner_number": owner_number
            },
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

        # Step 5: Respond back
        twilio_response = MessagingResponse()
        twilio_response.message(reply_text)
        return Response(content=str(twilio_response), media_type="application/xml")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        twilio_response = MessagingResponse()
        twilio_response.message("Sorry, something went wrong.")
        return Response(content=str(twilio_response), media_type="application/xml")
