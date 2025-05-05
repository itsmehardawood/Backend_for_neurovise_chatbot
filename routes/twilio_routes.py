# # from fastapi import APIRouter, Request, Form, Response
# # from twilio.twiml.messaging_response import MessagingResponse
# # from pymongo import MongoClient
# # from utils.utils import get_chat_completion, detect_scheduling_intent, create_calendar_event, EventRequest
# # from bson import ObjectId
# # import os
# # import datetime
# # import json
# # import re
# # import random
# # from dotenv import load_dotenv
# # from openai import OpenAI

# # load_dotenv()

# # twilio_router = APIRouter()

# # # MongoDB connection
# # mongo_uri = os.getenv("MONGO_URI")
# # client = MongoClient(mongo_uri)
# # db = client.Echo_db

# # users_collection = db.users
# # business_settings_collection = db.services
# # chat_history_collection = db.chat_history

# # # List of friendly emojis to use after greetings
# # GREETING_EMOJIS = ["👋", "😊", "👍", "✨", "🌟", "🙂", "👏", "🤗"]


# # def is_service_inquiry(text):
# #     service_keywords = [
# #         "services", "service", "offer", "providing", "do you have", 
# #         "what do you do", "what do you provide", "options", 
# #         "what are your", "tell me about your", "tell me what"
# #     ]
# #     text_lower = text.lower()
# #     return any(keyword in text_lower for keyword in service_keywords)


# # def is_greeting(text):
# #     greetings = ["hi", "hello", "hey", "hi there", "yo", "good morning", "good evening", "good afternoon"]
# #     text = text.lower().strip()
# #     return any(re.fullmatch(rf"{greet}", text) for greet in greetings)


# # def add_greeting_emoji(text, is_greeting_message=False):
# #     """Add emoji to greeting if needed."""
# #     if not is_greeting_message:
# #         return text
        
# #     # Check if the text already contains an emoji
# #     emoji_pattern = re.compile("["
# #         u"\U0001F600-\U0001F64F"  # emoticons
# #         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
# #         u"\U0001F680-\U0001F6FF"  # transport & map symbols
# #         u"\U0001F700-\U0001F77F"  # alchemical symbols
# #         u"\U0001F780-\U0001F7FF"  # Geometric Shapes
# #         u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
# #         u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
# #         u"\U0001FA00-\U0001FA6F"  # Chess Symbols
# #         u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
# #         u"\U00002702-\U000027B0"  # Dingbats
# #         u"\U000024C2-\U0001F251"
# #         "]+", flags=re.UNICODE)
    
# #     has_emoji = bool(emoji_pattern.search(text))
    
# #     if has_emoji:
# #         return text
    
# #     # If no emoji, check if it starts with a greeting
# #     greeting_pattern = re.compile(r'^(hello|hi|hey|greetings)(\s+there)?[,.!]?\s*', re.IGNORECASE)
# #     greeting_match = greeting_pattern.search(text)
    
# #     if greeting_match:
# #         # Extract the greeting
# #         greeting = greeting_match.group(0).rstrip()
# #         rest_of_text = text[len(greeting_match.group(0)):]
        
# #         # Add emoji to the greeting
# #         emoji = random.choice(GREETING_EMOJIS)
# #         text = f"{greeting}{emoji} {rest_of_text}"
# #     else:
# #         # No greeting found, add one
# #         emoji = random.choice(GREETING_EMOJIS)
# #         greetings = ["Hello", "Hi there", "Hey"]
# #         text = f"{random.choice(greetings)}{emoji} {text}"
    
# #     return text


# # @twilio_router.post("/twilio/whatsapp")
# # async def whatsapp_webhook(
# #     request: Request,
# #     Body: str = Form(...),
# #     From: str = Form(...),
# #     To: str = Form(...)
# # ):
# #     customer_number = From.replace("whatsapp:", "").replace("+", "").strip()
# #     owner_number = To.replace("whatsapp:", "").replace("+", "").strip()

# #     print(f"[DEBUG] Customer: {customer_number}, Owner: {owner_number}")

# #     owner_user = users_collection.find_one({"phone_number": owner_number})
# #     if not owner_user:
# #         print(f"[DEBUG] No matching owner user found.")
# #         return Response(status_code=200)

# #     owner_user_id = str(owner_user["_id"])

# #     business_settings = business_settings_collection.find_one({"user_id": ObjectId(owner_user_id)})
    
# #     # Updated prompt with clear instructions about emojis
# #     base_human_message = """You are a friendly person having a casual conversation.

# # IMPORTANT: You must sound like a real human, not a chatbot or AI assistant. 

# # AVOID completely:
# # - "How can I assist you today?"
# # - "I'm here to help"
# # - "How may I help you?"
# # - "I'd be happy to assist"
# # - "Is there anything else you need help with?"
# # - Any variation of these standard assistant phrases

# # DO THIS INSTEAD:
# # - Use casual, conversational language like "Got it"
# # - Talk like a friend would, not a service provider
# # - Use contractions (I'm, you're, we'll, that's)
# # - Be naturally helpful without saying you're "here to help"
# # - Be concise - humans don't write long, perfect paragraphs
# # - Occasionally use sentence fragments, like real texting
# # - Remember that less formal = more human sounding

# # Keep responses brief like a text message - not essay-length."""

# #     services_section = ""
# #     has_services = False

# #     if business_settings:
# #         custom_prompt = business_settings.get('system_prompt', '')

# #         if business_settings.get('services') and isinstance(business_settings.get('services'), list):
# #             valid_services = []
# #             for service in business_settings['services']:
# #                 if (isinstance(service, dict) and 
# #                     service.get('serviceName') and 
# #                     service.get('description') and 
# #                     service.get('isActive', False) == True):
# #                     valid_services.append(service)

# #             if valid_services:
# #                 has_services = True
# #                 services_info = "\n".join(
# #                     f"Service: {s['serviceName']}\nDescription: {s['description']}\nPrice: {s.get('price', 'N/A')}"
# #                     for s in valid_services
# #                 )
# #                 services_section = f"Available Services:\n{services_info}"
# #                 print(f"[DEBUG] Found {len(valid_services)} active services")

# #         tone = business_settings.get('chat_tone', 'professional')

# #         if services_section:
# #             system_message = f"{base_human_message}\n\n{custom_prompt}\n\n{services_section}\n\nRespond in a {tone}, human-like tone."
# #         else:
# #             system_message = f"{base_human_message}\n\n{custom_prompt}\n\nRespond in a {tone}, human-like tone."
# #     else:
# #         system_message = f"""{base_human_message}

# # For appointment scheduling, please collect:
# # 1. Preferred date
# # 2. Preferred time
# # 3. Contact email
# # 4. Service interested in
# # 5. Any special requests

# # Keep it conversational and friendly - like texting a colleague."""

# #     try:
# #         user_greeting = is_greeting(Body)
        
# #         if is_service_inquiry(Body):
# #             print("[DEBUG] Service inquiry detected.")
# #             if has_services:
# #                 enhanced_system_message = system_message + "\n\nThe user is asking about services. Talk about the available services in a friendly way."
# #                 reply_text = get_chat_completion(Body, enhanced_system_message)
# #             else:
# #                 no_services_message = base_human_message + "\n\nThe user is asking about services, but you don't have any active services configured yet. Politely explain that you're still setting up your services and will have more information soon. Suggest they can schedule a consultation to discuss their needs if they'd like."
# #                 reply_text = get_chat_completion(Body, no_services_message)

# #         elif user_greeting:
# #             print("[DEBUG] Greeting detected.")
# #             # Add special instruction for greeting responses
# #             greeting_system_message = system_message + "\n\nThe user has sent a greeting. Reply with a friendly greeting that includes an emoji."
# #             reply_text = get_chat_completion(Body, greeting_system_message)
# #             # Ensure emoji is present in greeting response
# #             reply_text = add_greeting_emoji(reply_text, True)

# #         else:
# #             is_scheduling = await detect_scheduling_intent(Body)

# #             if is_scheduling:
# #                 print("[DEBUG] Scheduling intent detected.")

# #                 client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# #                 response = client.chat.completions.create(
# #                     model="gpt-4-turbo",
# #                     messages=[
# #                         {"role": "system", "content": system_message},
# #                         {"role": "user", "content": Body}
# #                     ],
# #                     tools=[{
# #                         "type": "function",
# #                         "function": {
# #                             "name": "create_calendar_event",
# #                             "description": "Schedule a calendar appointment",
# #                             "parameters": {
# #                                 "type": "object",
# #                                 "properties": {
# #                                     "summary": {"type": "string"},
# #                                     "start_datetime": {"type": "string"},
# #                                     "end_datetime": {"type": "string"},
# #                                     "description": {"type": "string"},
# #                                     "attendees": {
# #                                         "type": "array",
# #                                         "items": {"type": "string", "format": "email"}
# #                                     }
# #                                 },
# #                                 "required": ["summary", "start_datetime", "end_datetime"]
# #                             }
# #                         }
# #                     }]
# #                 )

# #                 message = response.choices[0].message

# #                 if message.tool_calls:
# #                     try:
# #                         event_data = json.loads(message.tool_calls[0].function.arguments)
# #                         event_data['summary'] = event_data.get('summary') or "Appointment"

# #                         if owner_user.get('email'):
# #                             event_data.setdefault('attendees', []).append(owner_user['email'])

# #                         event_result = create_calendar_event(EventRequest(**event_data))

# #                         if event_result["status"] == "success":
# #                             reply_text = (
# #                                 f"All set! {event_data['summary']} scheduled for {event_data['start_datetime'].split('T')[0]} at {event_data['start_datetime'].split('T')[1][:5]}"
# #                             )
# #                             if event_result.get('meet_link'):
# #                                 reply_text += f"\n\nHere's your meeting link: {event_result['meet_link']}"
# #                         else:
# #                             reply_text = f"Hmm, couldn't schedule that. {event_result.get('message', 'Try again?')}"
# #                     except Exception as e:
# #                         print(f"[ERROR] Failed parsing event details: {e}")
# #                         reply_text = "Need a bit more info for scheduling. What date and time works for you? And your email?"
# #                 else:
# #                     reply_text = "Just need a few more details to get you scheduled. What date/time works? And what's your email?"
# #             else:
# #                 print("[DEBUG] Regular chat.")
# #                 reply_text = get_chat_completion(Body, system_message)

# #         # IMPORTANT: Only remove problematic greetings, NOT emojis
# #         reply_text = re.sub(r'^(hello there|hey there|hi there)[,.!]?\s+', '', reply_text, flags=re.IGNORECASE).strip()
        
# #         # DO NOT remove emojis - this was causing the issue!
# #         # Removed emoji_pattern and emoji removal code

# #         chat_history_collection.update_one(
# #             {"customer_number": customer_number, "owner_number": owner_number},
# #             {
# #                 "$setOnInsert": {
# #                     "customer_number": customer_number,
# #                     "owner_number": owner_number,
# #                     "name": "WhatsApp user",
# #                     "owner_user_id": owner_user_id
# #                 },
# #                 "$push": {
# #                     "messages": {
# #                         "timestamp": datetime.datetime.utcnow(),
# #                         "query": Body,
# #                         "response": reply_text
# #                     }
# #                 }
# #             },
# #             upsert=True
# #         )

# #         twilio_response = MessagingResponse()
# #         twilio_response.message(reply_text)
# #         return Response(content=str(twilio_response), media_type="application/xml")

# #     except Exception as e:
# #         print(f"[ERROR] {str(e)}")
# #         twilio_response = MessagingResponse()
# #         twilio_response.message("Oops! Something went wrong on our end. Mind trying again?")
# #         return Response(content=str(twilio_response), media_type="application/xml")



# from json import encoder
# import json
# import uuid
# from bson import ObjectId
# from fastapi import APIRouter, Request, Form, Response
# from twilio.twiml.messaging_response import MessagingResponse
# from utils.utils import get_chat_completion
# import os
# import traceback
# from dotenv import load_dotenv
# from src.constant import system_message

# from fastapi import APIRouter, Request, Form, Response
# from twilio.twiml.messaging_response import MessagingResponse
# from utils.utils import get_chat_completion
# from pymongo import MongoClient
# import os
# from dotenv import load_dotenv
# from src.constant import system_message
# import datetime

# from motor.motor_asyncio import AsyncIOMotorClient
# import ssl
# MONGO_URI=os.getenv("MONGO_URI")

# client = AsyncIOMotorClient(
#     MONGO_URI,
#     tls=True
# )

# db = client.Echo_db
# users_collection = db.users
# business_settings_collection = db.services
# chat_history_collection = db.chat_history
# chat_sessions_collection = db.chat_sessions

# twilio_router = APIRouter()
# # MongoDB connection
# # mongo_uri = os.getenv("MONGO_URI")
# # client = MongoClient(mongo_uri)
# # db = client["Echo_db"]
# # users_collection = db["users"]
# # chat_collection = db["whatsapp_chat"]
# # chat_sessions_collection = db["chat_sessions"]

# import openai
# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv(override=True)  # Load environment variables from .env file

# openai.api_key = os.getenv("OPENAI_API_KEY")
# model = os.getenv("OPENAI_MODEL")  # Make sure OPENAI_MODEL is set in your .env
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def get_chat_completion1(user_query: str, system_message: str) -> str:
#     print("system message:", system_message)
#     print("here is the query:", user_query)
#     try:
#         # Create a completion using the OpenAI API
#         print(f"Querying OpenAI with: {user_query}")
#         completion = openai.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": user_query}
#             ]
#         )
#         print(f"OpenAI Completion Response: {completion}")  # Log the response

#         # Return the content of the chat response
#         return completion.choices[0].message.content

#     except Exception as e:
#         # Handle any potential errors, such as connection issues or invalid responses
#         print(f"Error during OpenAI API call: {e}")
#         return "Sorry, there was an error processing your request."
    
    
    
    
    
# # @twilio_router.post("/twilio/whatsapp")
# # async def whatsapp_webhook(
# #     request: Request,
# #     Body: str = Form(...),
# #     From: str = Form(...)
# # ):
# #     # Normalize phone number (remove "whatsapp:" and "+")
# #     phone_number = From.replace("whatsapp:", "").replace("+", "").strip()
# #     print(f"[DEBUG] Looking for number: {repr(phone_number)}")

# #     # Find user
# #     user = phone_number

# #     if user:

# #         # Generate chatbot reply
# #         from src.constant import system_message

# #         reply = get_chat_completion1(Body, system_message)

# #         # Twilio response
# #         twilio_response = MessagingResponse()
# #         twilio_response.message(reply)
# #         return Response(content=str(twilio_response), media_type="application/xml")
# #     else:
# #         print(f"[DEBUG] No matching user found for phone number: {phone_number}")
# #         return Response(status_code=200)
    
# from fastapi import HTTPException
# from bson import ObjectId
# from datetime import datetime
# from fastapi.responses import JSONResponse
# import os
# import json
# import uuid
# import traceback
# from datetime import datetime
# from typing import List, Optional
# from fastapi import HTTPException, Request
# from bson import ObjectId
# from pydantic import BaseModel, EmailStr
# from openai import OpenAI
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from fastapi import Request as FastAPIRequest
# from google.auth.transport.requests import Request as GoogleAuthRequest


# # Initialize clients
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# google_credentials_path = 'credentials.json'
# google_token_path = 'token.json'
# google_scopes = ['https://www.googleapis.com/auth/calendar.events']

# class EventRequest(BaseModel):
#     summary: str
#     start_datetime: str
#     end_datetime: str
#     description: Optional[str] = None
#     attendees: Optional[List[EmailStr]] = None


# import tiktoken
# from typing import List, Dict, Any, Optional

# # Add this at the top of your file with other imports
# encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

# def count_tokens(messages: List[Dict[str, Any]]) -> int:
#     """Count the number of tokens in a list of messages."""
#     num_tokens = 0
#     for message in messages:
#         # Count tokens in the content
#         content = message.get("content", "")
#         num_tokens += len(encoder.encode(content))
        
#         # Add tokens for message metadata
#         num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        
#         # Count tokens in the role
#         role = message.get("role", "")
#         num_tokens += len(encoder.encode(role))
        
#         # Count tokens in function calls if present
#         if "tool_calls" in message:
#             for tool_call in message["tool_calls"]:
#                 if "function" in tool_call:
#                     function = tool_call["function"]
#                     # Add tokens for function name and arguments
#                     if "name" in function:
#                         num_tokens += len(encoder.encode(function["name"]))
#                     if "arguments" in function:
#                         num_tokens += len(encoder.encode(function["arguments"]))
    
#     num_tokens += 3  # every reply is primed with <im_start>assistant
#     return num_tokens

# def truncate_system_message(system_message: str, max_tokens: int) -> str:
#     """Truncate system message to fit within max_tokens."""
#     tokens = encoder.encode(system_message)
#     if len(tokens) <= max_tokens:
#         return system_message
    
#     # Truncate and add a note about truncation
#     truncated_tokens = tokens[:max_tokens - 10]  # Leave space for truncation note
#     truncated_message = encoder.decode(truncated_tokens)
#     return truncated_message + "\n[Content truncated due to length]"

# def prepare_business_info(business_settings: dict, max_tokens: int = 4000) -> str:
#     """Prepare business information with token limits in mind, filtering for active services only."""
#     # Start with the base system prompt
#     system_message = business_settings.get('system_prompt', '')
    
#     # If services exist, add them in a condensed format, but ONLY active ones
#     if business_settings.get('services'):
#         # Filter for active services only
#         active_services = [s for s in business_settings['services'] if s.get('isActive', True)]
        
#         # Limit the number of services if needed
#         max_services = 5  # Adjust based on your needs
#         services = active_services[:max_services]
        
#         if services:  # Only proceed if there are active services
#             services_info = "\n".join(
#                 f"- {s['serviceName']}: {s.get('price', 'N/A')} - {s.get('description', '')[:100]}..." 
#                 for s in services
#             )
            
#             system_message = f"Available Services:\n{services_info}\n\n{system_message}"
            
#             # Add instruction to only discuss active services
#             system_message += "\n\nIMPORTANT: ONLY discuss and recommend the services listed above. Do NOT mention any other services."
    
#     # Add tone instruction
#     tone = business_settings.get('chat_tone', 'professional')
#     system_message = f"{system_message}\n\nRespond in a {tone} tone."
    
#     # Ensure the system message fits within token limit
#     return truncate_system_message(system_message, max_tokens)


# async def detect_scheduling_intent(query: str) -> bool:
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{
#             "role": "system",
#             "content": "Determine if the user wants to schedule something. Reply only 'yes' or 'no'."
#         }, {
#             "role": "user",
#             "content": query
#         }],
#         temperature=0
#     )
#     return response.choices[0].message.content.strip().lower() == 'yes'

# async def save_chat_to_db(session_id: str, query: str, response: str, is_scheduling: bool, event_details: Optional[dict] = None):
#     """Helper function to save chat history to database"""
#     try:
#         update_data = {
#             "$push": {
#                 "messages": {
#                     "query": query,
#                     "response": response,
#                     "timestamp": datetime.utcnow(),
#                     "is_scheduling": is_scheduling
#                 }
#             },
#             "$set": {"last_activity": datetime.utcnow()}
#         }
        
#         if event_details:
#             update_data["$push"]["messages"]["event_details"] = event_details
        
#         await chat_sessions_collection.update_one(
#             {"session_id": session_id},
#             update_data
#         )
#     except Exception as e:
#         print(f"Failed to save chat to database: {str(e)}")
#         traceback.print_exc()
        
    
# def get_calendar_service():
#     creds = None
#     if os.path.exists(google_token_path):
#         creds = Credentials.from_authorized_user_file(google_token_path, google_scopes)
    
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(GoogleAuthRequest())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(google_credentials_path, google_scopes)
#             creds = flow.run_local_server(port=0)
        
#         with open(google_token_path, 'w') as token:
#             token.write(creds.to_json())
    
#     return build('calendar', 'v3', credentials=creds)
    
# def create_calendar_event(event: EventRequest):
#     service = get_calendar_service()
    
#     event_body = {
#         'summary': event.summary,
#         'description': event.description,
#         'start': {'dateTime': event.start_datetime, 'timeZone': 'UTC'},
#         'end': {'dateTime': event.end_datetime, 'timeZone': 'UTC'},
#         'attendees': [{'email': email} for email in event.attendees] if event.attendees else [],
#         'conferenceData': {
#             'createRequest': {
#                 'requestId': str(uuid.uuid4()),
#                 'conferenceSolutionKey': {'type': 'hangoutsMeet'}
#             }
#         }
#     }

#     try:
#         created_event = service.events().insert(
#             calendarId='primary',
#             body=event_body,
#             conferenceDataVersion=1
#         ).execute()

#         return {
#             "status": "success",
#             "event_id": created_event.get('id'),
#             "htmlLink": created_event.get('htmlLink'),
#             "meet_link": created_event.get('hangoutLink')
#         }
#     except HttpError as error:
#         return {"status": "error", "message": str(error)}

# @twilio_router.post("/twilio/whatsapp")
# async def whatsapp_webhook(
#     request: Request,
#     Body: str = Form(...),
#     From: str = Form(...)
# ):
#     try:
#         # Normalize phone number (remove "whatsapp:" and "+")
#         phone_number = From.replace("whatsapp:", "").replace("+", "").strip()
#         print(f"[DEBUG] Looking for number: {repr(phone_number)}")

#         # Find or create session for this number
#         session = await chat_sessions_collection.find_one({"phone_number": phone_number})
#         if not session:
#             # Create new session for this number
#             session_id = str(uuid.uuid4())
#             await chat_sessions_collection.insert_one({
#                 "session_id": session_id,
#                 "phone_number": phone_number,
#                 "messages": [],
#                 "created_at": datetime.utcnow(),
#                 "last_activity": datetime.utcnow()
#             })
#             session = await chat_sessions_collection.find_one({"phone_number": phone_number})
        
#         # Find the user by phone number
#         user = await users_collection.find_one({"phone": phone_number})
#         if not user:
#             # If no matching user, check if this is a registration request
#             is_registration = await detect_registration_intent(Body)
#             if is_registration:
#                 # Handle registration process
#                 registration_response = "Welcome! To register, please reply with your name and email in this format: 'Register: [Your Name], [Your Email]'"
                
#                 # Twilio response
#                 twilio_response = MessagingResponse()
#                 twilio_response.message(registration_response)
#                 return Response(content=str(twilio_response), media_type="application/xml")
#             else:
#                 # Create temporary user for this session
#                 user_id = str(ObjectId())
#                 user = {"_id": ObjectId(user_id), "phone": phone_number}
        
#         user_id = str(user.get("_id"))
        
#         # DEBUG: Log user query
#         print(f"Processing WhatsApp request. Query: {Body}")
        
#         # Get business settings
#         business_settings = await business_settings_collection.find_one({"user_id": ObjectId(user_id)})
        
#         # Filter for active services
#         active_services = []
#         if business_settings and business_settings.get('services'):
#             active_services = [s for s in business_settings['services'] if s.get('isActive', True)]
#             print(f"Found {len(active_services)} active services out of {len(business_settings.get('services', []))} total services")
        
#         # Prepare system message - prioritize business settings
#         if business_settings:
#             # Add debugging info about business settings
#             print(f"Found business settings: {business_settings.get('name', 'Unnamed Business')}")
            
#             # Use the token-aware function to prepare business info
#             system_message = prepare_business_info(business_settings)
            
#             # Enhanced system prompt for better appointment handling
#             system_message += "\n\nIMPORTANT: You are a virtual assistant on WhatsApp that helps with booking appointments. If a user wants to schedule an appointment, help them by collecting the necessary information and use the provided function to create a calendar event. ONLY discuss and recommend ACTIVE services listed above."
#         else:
#             # Default system message when no business settings exist
#             system_message = """You are a helpful WhatsApp assistant that manages appointments. 
# For appointment scheduling, please collect:
# 1. Preferred date
# 2. Preferred time
# 3. Contact email
# 4. Service interested in
# 5. Any special requests

# IMPORTANT: You CAN and SHOULD book appointments when requested. Use the appointment scheduling function when appropriate.
# Respond in a professional tone. Keep responses concise as this is WhatsApp."""

#         # Set max tokens limit with buffer
#         MAX_TOKENS = 8000  # Lower for WhatsApp to ensure messages aren't too long
        
#         # DEBUG: Log system message beginning
#         print(f"System message starts with: {system_message[:100]}...")
        
#         # Determine intent and process
#         is_scheduling = await detect_scheduling_intent(Body)
        
#         # DEBUG: Log scheduling intent
#         print(f"Scheduling intent detected: {is_scheduling}")
        
#         if is_scheduling:
#             # DEBUG: Log entering scheduling flow
#             print("Entering scheduling flow...")
            
#             # Enhanced system message specific for scheduling
#             scheduling_system_message = system_message
            
#             # Add active services info explicitly for scheduling
#             if active_services:
#                 active_services_info = "\n".join(
#                     f"- {s['serviceName']}: {s.get('price', 'N/A')}" 
#                     for s in active_services[:5]  # Limit to 5 services
#                 )
#                 scheduling_system_message += f"\n\nONLY offer these active services for booking:\n{active_services_info}"
#             else:
#                 scheduling_system_message += "\n\nThere are no specific services defined. Schedule a general appointment."
            
#             scheduling_system_message += "\n\nThe user is trying to schedule an appointment via WhatsApp. Help them by collecting all necessary information and use the create_calendar_event function to book it. DO NOT refuse to book appointments - that is your primary function. ONLY discuss and recommend ACTIVE services. Keep messages concise for WhatsApp."
            
#             # Define the function call with calendar event schema
#             calendar_function = {
#                 "type": "function",
#                 "function": {
#                     "name": "create_calendar_event",
#                     "description": "Schedule a calendar appointment",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "summary": {"type": "string", "description": "Title of the appointment"},
#                             "start_datetime": {"type": "string", "description": "Start time in ISO format (YYYY-MM-DDTHH:MM:SS)"},
#                             "end_datetime": {"type": "string", "description": "End time in ISO format (YYYY-MM-DDTHH:MM:SS)"},
#                             "description": {"type": "string", "description": "Additional details about the appointment"},
#                             "attendees": {
#                                 "type": "array",
#                                 "items": {"type": "string", "format": "email"},
#                                 "description": "List of email addresses for attendees"
#                             }
#                         },
#                         "required": ["summary", "start_datetime", "end_datetime"]
#                     }
#                 }
#             }
            
#             # Retrieve the conversation history
#             conversation_history = []
#             if session and "messages" in session:
#                 # Get the last 5 messages to provide context
#                 for msg in session["messages"][-5:]:
#                     conversation_history.append({"role": "user", "content": msg.get("query", "")})
#                     conversation_history.append({"role": "assistant", "content": msg.get("response", "")})
            
#             # Create messages array with history
#             messages = [
#                 {"role": "system", "content": scheduling_system_message},
#             ]
            
#             # Add conversation history if available
#             if conversation_history:
#                 messages.extend(conversation_history)
            
#             # Add current user message
#             messages.append({"role": "user", "content": Body})
            
#             # Check if token count exceeds limit
#             message_tokens = count_tokens(messages)
#             if message_tokens > MAX_TOKENS:
#                 # If too long, skip history and just use current message
#                 messages = [
#                     {"role": "system", "content": scheduling_system_message},
#                     {"role": "user", "content": Body}
#                 ]
                
#                 # If still too long, truncate system message
#                 message_tokens = count_tokens(messages)
#                 if message_tokens > MAX_TOKENS:
#                     excess_tokens = message_tokens - MAX_TOKENS + 500  # Buffer
#                     max_system_tokens = len(encoder.encode(scheduling_system_message)) - excess_tokens
#                     if max_system_tokens < 500:  # Minimum viable system message
#                         max_system_tokens = 500
                    
#                     scheduling_system_message = truncate_system_message(scheduling_system_message, max_system_tokens)
#                     messages[0]["content"] = scheduling_system_message
            
#             # DEBUG: Log messages before API call
#             print(f"Making scheduling API call with message tokens: {count_tokens(messages)}")
            
#             # Make the API call with token-managed messages
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=messages,
#                 tools=[calendar_function],
#                 tool_choice="auto"  # Explicitly tell the model to use tools when appropriate
#             )

#             # DEBUG: Log basic response info
#             print(f"Received response. Contains tool calls: {hasattr(response.choices[0].message, 'tool_calls') and bool(response.choices[0].message.tool_calls)}")
            
#             message = response.choices[0].message
            
#             # Check for tool calls from the API response
#             if message.tool_calls:
#                 try:
#                     # DEBUG: Log tool call details
#                     print(f"Tool call received: {message.tool_calls[0].function.name}")
                    
#                     event_data = json.loads(message.tool_calls[0].function.arguments)
#                     print(f"Event data: {event_data}")
                    
#                     # Validate that the requested service is active (if service is specified)
#                     is_valid_service = True
#                     if event_data.get('description') and business_settings and business_settings.get('services'):
#                         # Basic check if the description contains an inactive service name
#                         inactive_service_names = [s['serviceName'].lower() for s in business_settings.get('services', []) 
#                                             if not s.get('isActive', True)]
                        
#                         for inactive_service in inactive_service_names:
#                             if inactive_service in event_data.get('description', '').lower():
#                                 is_valid_service = False
#                                 break
                    
#                     if not is_valid_service:
#                         # If inactive service detected, return a helpful message
#                         inactive_response = "I notice you're interested in a service that isn't currently available. Here are the services we currently offer:\n\n"
#                         inactive_response += "\n".join(
#                             f"- {s['serviceName']}: {s.get('price', 'N/A')}" 
#                             for s in active_services
#                         )
#                         inactive_response += "\n\nWould you like to book an appointment for one of these services instead?"
                        
#                         await save_chat_to_db(
#                             session_id=session["session_id"],
#                             query=Body,
#                             response=inactive_response,
#                             is_scheduling=True
#                         )
                        
#                         # Twilio response
#                         twilio_response = MessagingResponse()
#                         twilio_response.message(inactive_response)
#                         return Response(content=str(twilio_response), media_type="application/xml")
                    
#                     event_data['summary'] = event_data.get('summary') or "Appointment"
                    
#                     # Make sure we have both attendees and user email if available
#                     if 'attendees' not in event_data:
#                         event_data['attendees'] = []
                    
#                     if user.get('email') and user['email'] not in event_data['attendees']:
#                         event_data['attendees'].append(user['email'])
                    
#                     # Create the calendar event
#                     event_result = create_calendar_event(EventRequest(**event_data))
                    
#                     if event_result["status"] == "success":
#                         # Format a friendly response for successful scheduling
#                         start_datetime = event_data['start_datetime'].replace('T', ' at ').split('+')[0]
                        
#                         response_text = f"Great! I've scheduled your appointment:\n\n"
#                         response_text += f"📅 {event_data['summary']}\n"
#                         response_text += f"🕒 {start_datetime}\n"
                        
#                         if event_data.get('description'):
#                             response_text += f"📝 {event_data['description']}\n"
                        
#                         if event_result.get('meet_link'):
#                             response_text += f"\n🔗 Video call link: {event_result['meet_link']}"
                        
#                         # Add confirmation number
#                         response_text += f"\n\nConfirmation #: {event_result['event_id'][-6:]}"
                        
#                         # Save to database
#                         await save_chat_to_db(
#                             session_id=session["session_id"],
#                             query=Body,
#                             response=response_text,
#                             is_scheduling=True,
#                             event_details=event_result
#                         )
                        
#                         # Twilio response
#                         twilio_response = MessagingResponse()
#                         twilio_response.message(response_text)
#                         return Response(content=str(twilio_response), media_type="application/xml")
#                     else:
#                         error_response = f"I tried to schedule your appointment, but encountered an error: {event_result.get('message')}"
#                         await save_chat_to_db(
#                             session_id=session["session_id"],
#                             query=Body,
#                             response=error_response,
#                             is_scheduling=True
#                         )
                        
#                         # Twilio response
#                         twilio_response = MessagingResponse()
#                         twilio_response.message(error_response)
#                         return Response(content=str(twilio_response), media_type="application/xml")
#                 except Exception as tool_error:
#                     # Log the specific error with the tool call
#                     print(f"Error processing tool call: {str(tool_error)}")
#                     traceback.print_exc()
                    
#                     # Fall back to regular response
#                     fallback_response = f"I encountered an issue while trying to schedule your appointment. Please try again with complete details including date, time, and purpose."
                    
#                     await save_chat_to_db(
#                         session_id=session["session_id"],
#                         query=Body,
#                         response=fallback_response,
#                         is_scheduling=True
#                     )
                    
#                     # Twilio response
#                     twilio_response = MessagingResponse()
#                     twilio_response.message(fallback_response)
#                     return Response(content=str(twilio_response), media_type="application/xml")
#             else:
#                 # The model didn't use the function, but we know it's a scheduling intent
#                 print("Model didn't use tool call despite scheduling intent. Using regular chat response.")
                
#                 # Modified system message that encourages getting the details for next time
#                 assist_system_message = system_message + "\n\nThe user wants to schedule an appointment via WhatsApp. If you don't have enough details yet, ask for specific information like date, time, and purpose. DO NOT refuse to help with scheduling - that is your primary purpose. Never say you cannot book appointments. ONLY mention active services. Keep responses concise for WhatsApp."
                
#                 if active_services:
#                     active_services_info = "\n".join(
#                         f"- {s['serviceName']}: {s.get('price', 'N/A')}" 
#                         for s in active_services[:5]
#                     )
#                     assist_system_message += f"\n\nACTIVE SERVICES:\n{active_services_info}"
                
#                 messages = [
#                     {"role": "system", "content": assist_system_message},
#                     {"role": "user", "content": Body}
#                 ]
                
#                 # Check token count for regular messages
#                 if count_tokens(messages) > MAX_TOKENS:
#                     # Apply token reduction
#                     assist_system_message = truncate_system_message(assist_system_message, 2000)
#                     messages[0]["content"] = assist_system_message
                
#                 # Use regular chat completion to ask for more details
#                 chat_completion = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=messages
#                 )
                
#                 # Get text response
#                 chat_response = chat_completion.choices[0].message.content
                
#                 # Save to database
#                 await save_chat_to_db(
#                     session_id=session["session_id"],
#                     query=Body,
#                     response=chat_response,
#                     is_scheduling=True
#                 )
                
#                 # Twilio response
#                 twilio_response = MessagingResponse()
#                 twilio_response.message(chat_response)
#                 return Response(content=str(twilio_response), media_type="application/xml")
        
#         # Regular chat flow (non-scheduling intent)
#         print("Using regular WhatsApp chat flow...")
        
#         # Check if we should add greeting formatting
#         has_greeting = check_for_greeting(Body)
        
#         # For WhatsApp, we want responses to be concise
#         system_message += "\n\nIMPORTANT: Keep your responses concise and to the point for WhatsApp. Use emojis sparingly."
        
#         # Create messages for regular chat flow
#         messages = [
#             {"role": "system", "content": system_message},
#             {"role": "user", "content": Body}
#         ]
        
#         # Check token count for regular messages
#         message_tokens = count_tokens(messages)
#         if message_tokens > MAX_TOKENS:
#             # Apply token reduction logic
#             excess_tokens = message_tokens - MAX_TOKENS + 500  # Buffer
#             max_system_tokens = len(encoder.encode(system_message)) - excess_tokens
#             if max_system_tokens < 500:
#                 max_system_tokens = 500
            
#             system_message = truncate_system_message(system_message, max_system_tokens)
#             messages[0]["content"] = system_message
        
#         # Make regular chat API call
#         chat_completion = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=messages
#         )
        
#         # Get the response content
#         chat_response = chat_completion.choices[0].message.content
        
#         # Save to database
#         await save_chat_to_db(
#             session_id=session["session_id"],
#             query=Body,
#             response=chat_response,
#             is_scheduling=is_scheduling
#         )

#         # Twilio response
#         twilio_response = MessagingResponse()
#         twilio_response.message(chat_response)
#         return Response(content=str(twilio_response), media_type="application/xml")

#     except Exception as e:
#         print(f"Error in WhatsApp webhook: {str(e)}")
#         traceback.print_exc()
        
#         # Send a friendly error message to user
#         error_response = "I'm sorry, I encountered an error processing your request. Please try again later."
#         twilio_response = MessagingResponse()
#         twilio_response.message(error_response)
#         return Response(content=str(twilio_response), media_type="application/xml")


# # Add these new helper functions for WhatsApp specific needs
# async def detect_registration_intent(query: str) -> bool:
#     """Detect if the user wants to register"""
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{
#             "role": "system",
#             "content": "Determine if the user wants to register or create an account. Reply only 'yes' or 'no'."
#         }, {
#             "role": "user",
#             "content": query
#         }],
#         temperature=0
#     )
#     return response.choices[0].message.content.strip().lower() == 'yes'


# def check_for_greeting(query: str) -> bool:
#     """Check if the message is a greeting"""
#     greeting_words = [
#         "hi", "hello", "hey", "greetings", "good morning", 
#         "good afternoon", "good evening", "howdy", "hola", "salut"
#     ]
    
#     # Convert to lowercase
#     query_lower = query.lower().strip()
    
#     # Check if query starts with a greeting word
#     for greeting in greeting_words:
#         if query_lower.startswith(greeting):
#             return True
            
#     # Check if query is just a greeting (allowing for punctuation)
#     clean_query = query_lower.rstrip('!.,?')
#     return clean_query in greeting_words
 
 
 
 
 
 
import datetime
import json
import os
import random
import re
import traceback
import uuid
from typing import List, Optional

import tiktoken
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, Form, Request, Response
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient
from twilio.twiml.messaging_response import MessagingResponse

from utils.utils import get_chat_completion, EventRequest

load_dotenv()

twilio_router = APIRouter()

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client.Echo_db

users_collection = db.users
business_settings_collection = db.services
chat_history_collection = db.chat_history

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
google_credentials_path = 'credentials.json'
google_token_path = 'token.json'
google_scopes = ['https://www.googleapis.com/auth/calendar.events']

# List of friendly emojis to use after greetings
GREETING_EMOJIS = ["👋", "😊", "👍", "✨", "🌟", "🙂", "👏", "🤗"]

# Initialize tokenizer for token counting
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(messages: List[dict]) -> int:
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

def is_greeting(text):
    """Check if a message is a simple greeting."""
    greetings = ["hi", "hello", "hey", "hi there", "yo", "good morning", "good evening", "good afternoon"]
    text = text.lower().strip()
    return any(re.fullmatch(rf"{greet}", text) for greet in greetings)

def is_service_inquiry(text):
    """Check if a message is asking about services."""
    service_keywords = [
        "services", "service", "offer", "providing", "do you have", 
        "what do you do", "what do you provide", "options", 
        "what are your", "tell me about your", "tell me what"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in service_keywords)

def add_greeting_emoji(text, is_greeting_message=False):
    """Add emoji to greeting if needed."""
    if not is_greeting_message:
        return text
    
    # For debugging
    print(f"[DEBUG] Adding emoji to greeting. Original text: {text[:30]}...")
        
    # Check if the text already contains an emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    has_emoji = bool(emoji_pattern.search(text))
    
    if has_emoji:
        print("[DEBUG] Text already has emoji, returning as is")
        return text
    
    # If no emoji, check if it starts with a greeting
    greeting_pattern = re.compile(r'^(hello|hi|hey|greetings|שלום|היי)(\s+there)?[,.!]?\s*', re.IGNORECASE)
    greeting_match = greeting_pattern.search(text)
    
    # Add Hebrew text pattern too
    hebrew_pattern = re.compile(r'^[\u0590-\u05FF\s,.!?]+', re.UNICODE)
    is_hebrew = hebrew_pattern.match(text)
    
    if greeting_match:
        # Extract the greeting
        greeting = greeting_match.group(0).rstrip()
        rest_of_text = text[len(greeting_match.group(0)):]
        
        # Add emoji to the greeting
        emoji = random.choice(GREETING_EMOJIS)
        modified_text = f"{greeting}{emoji} {rest_of_text}"
        print(f"[DEBUG] Modified existing greeting. New text: {modified_text[:30]}...")
        return modified_text
    else:
        # For Hebrew text or when no greeting found
        emoji = random.choice(GREETING_EMOJIS)
        
        # Choose appropriate greeting based on language
        if is_hebrew:
            greetings = ["שלום", "היי", "בוקר טוב"]
        else:
            greetings = ["Hello", "Hi", "Hey"]
            
        modified_text = f"{random.choice(greetings)}{emoji} {text}"
        print(f"[DEBUG] Added new greeting. New text: {modified_text[:30]}...")
        return modified_text

def get_calendar_service():
    """Initialize and return the Google Calendar service."""
    try:
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
    except Exception as e:
        print(f"[ERROR] Failed to initialize calendar service: {str(e)}")
        traceback.print_exc()
        raise

def create_calendar_event(event: EventRequest):
    """Create a Google Calendar event."""
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
    """Detect if a message is about scheduling an appointment."""
    response = openai_client.chat.completions.create(
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

def prepare_system_message(business_settings, is_greeting=False, is_service_inquiry=False, is_scheduling=False):
    """Prepare a system message with token management."""
    # Base human message - common for all scenarios
    base_human_message = """You are a friendly person having a casual conversation. By Default response in Hebrew language if user ask for english then response accordingly.

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

    # Default system message if no business settings are found
    if not business_settings:
        system_message = f"""{base_human_message}

For appointment scheduling, please collect:
1. Preferred date
2. Preferred time
3. Contact email
4. Service interested in
5. Any special requests

Keep it conversational and friendly - like texting a colleague."""
        return system_message

    # Extract business custom prompt
    custom_prompt = business_settings.get('system_prompt', '')
    
    # Get business tone
    tone = business_settings.get('chat_tone', 'professional')

    # Filter for active services only
    active_services = []
    if business_settings.get('services') and isinstance(business_settings.get('services'), list):
        active_services = [
            service for service in business_settings['services']
            if (isinstance(service, dict) and 
                service.get('serviceName') and 
                service.get('description') and 
                service.get('isActive', False) == True)
        ]

    # Create services information for the first 5 active services
    services_section = ""
    if active_services:
        services_info = "\n".join(
            f"Service: {s['serviceName']}\nDescription: {s['description'][:100]}...\nPrice: {s.get('price', 'N/A')}"
            for s in active_services[:5]  # Limit to 5 services to reduce token count
        )
        services_section = f"Available Services:\n{services_info}"
    
    # Build the system message with specific sections based on the type of inquiry
    system_message = f"{base_human_message}\n\n{custom_prompt}"
    
    if services_section and (is_service_inquiry or is_scheduling):
        system_message += f"\n\n{services_section}"
    
    # Add special instructions based on message type
    if is_greeting:
        system_message += "\n\nThe user has sent a greeting. Reply with a friendly greeting that includes an emoji."
    
    if is_service_inquiry:
        system_message += "\n\nThe user is asking about services. Talk about the available services in a friendly way."
    
    if is_scheduling:
        system_message += "\n\nThe user wants to schedule an appointment. Collect necessary information like date, time, and purpose."
    
    # Add tone instruction
    system_message += f"\n\nRespond in a {tone}, human-like tone."
    
    # Ensure the system message doesn't exceed token limit (4000 is a safe limit)
    return truncate_system_message(system_message, 4000)

@twilio_router.post("/twilio/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
):
    try:
        # Extract phone numbers
        customer_number = From.replace("whatsapp:", "").replace("+", "").strip()
        owner_number = To.replace("whatsapp:", "").replace("+", "").strip()

        print(f"[DEBUG] Customer: {customer_number}, Owner: {owner_number}")

        # Find the business owner
        owner_user = users_collection.find_one({"phone_number": owner_number})
        if not owner_user:
            print(f"[DEBUG] No matching owner user found.")
            return Response(status_code=200)

        owner_user_id = str(owner_user["_id"])

        # Get business settings
        business_settings = business_settings_collection.find_one({"user_id": ObjectId(owner_user_id)})
        
        # Count active services for logging
        active_service_count = 0
        if business_settings and business_settings.get('services'):
            active_service_count = sum(
                1 for s in business_settings['services'] 
                if isinstance(s, dict) and s.get('isActive', False) == True
            )
            print(f"[DEBUG] Found {active_service_count} active services")
        
        # Detect message intent
        user_greeting = is_greeting(Body)
        user_service_inquiry = is_service_inquiry(Body)
        
        print(f"[DEBUG] {'Greeting detected.' if user_greeting else ''}")
        print(f"[DEBUG] {'Service inquiry detected.' if user_service_inquiry else ''}")

        # Check for scheduling intent 
        # Only run this if it's not a service inquiry (to save API calls)
        is_scheduling = False
        if not user_service_inquiry and not user_greeting:
            try:
                is_scheduling = await detect_scheduling_intent(Body)
                if is_scheduling:
                    print("[DEBUG] Scheduling intent detected.")
            except Exception as e:
                print(f"[ERROR] Failed to detect scheduling intent: {str(e)}")
                # Continue even if scheduling detection fails

        # Prepare system message based on intent
        system_message = prepare_system_message(
            business_settings,
            is_greeting=user_greeting,
            is_service_inquiry=user_service_inquiry,
            is_scheduling=is_scheduling
        )

        # Handle scheduling intent
        if is_scheduling:
            try:
                # Create an optimized system message
                scheduling_system_message = system_message + "\n\nThe user is trying to schedule an appointment. Help the user schedule by collecting the necessary information and use the function if you have enough info."
                
                # Check token count for scheduling message
                scheduling_messages = [
                    {"role": "system", "content": scheduling_system_message},
                    {"role": "user", "content": Body}
                ]
                
                # Maximum context length for GPT-3.5-turbo
                MAX_TOKENS = 16000
                
                # Check if we need to truncate
                message_tokens = count_tokens(scheduling_messages)
                if message_tokens > MAX_TOKENS:
                    # Calculate how much to truncate
                    excess_tokens = message_tokens - MAX_TOKENS + 500  # Buffer
                    
                    # Truncate system message
                    max_system_tokens = len(encoder.encode(scheduling_system_message)) - excess_tokens
                    if max_system_tokens < 500:  # Minimum viable system message
                        max_system_tokens = 500
                    
                    scheduling_system_message = truncate_system_message(scheduling_system_message, max_system_tokens)
                    scheduling_messages[0]["content"] = scheduling_system_message
                
                # Create function configuration for calendar event
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
                
                # Make the API call
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=scheduling_messages,
                    tools=[calendar_function],
                    tool_choice="auto"
                )

                message = response.choices[0].message
                
                # Process tool calls if available
                if message.tool_calls:
                    try:
                        event_data = json.loads(message.tool_calls[0].function.arguments)
                        event_data['summary'] = event_data.get('summary') or "Appointment"

                        if owner_user.get('email'):
                            event_data.setdefault('attendees', []).append(owner_user['email'])

                        event_result = create_calendar_event(EventRequest(**event_data))

                        if event_result["status"] == "success":
                            reply_text = (
                                f"All set! {event_data['summary']} scheduled for {event_data['start_datetime'].split('T')[0]} at {event_data['start_datetime'].split('T')[1][:5]}"
                            )
                            if event_result.get('meet_link'):
                                reply_text += f"\n\nHere's your meeting link: {event_result['meet_link']}"
                        else:
                            reply_text = f"Hmm, couldn't schedule that. {event_result.get('message', 'Try again?')}"
                    except Exception as e:
                        print(f"[ERROR] Failed parsing event details: {e}")
                        reply_text = "Need a bit more info for scheduling. What date and time works for you? And your email?"
                else:
                    # If no tool calls, just use the message content
                    reply_text = message.content
            except Exception as e:
                print(f"[ERROR] Scheduling error: {str(e)}")
                reply_text = "I'd like to help schedule that, but I need a bit more information. Could you specify the date, time, and what service you're interested in?"
        else:
            # Handle regular queries
            try:
                print(f"[DEBUG] Processing regular chat. Query: {Body}")
                
                # Create messages for the API
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": Body}
                ]
                
                # Check token count
                MAX_TOKENS = 16000
                message_tokens = count_tokens(messages)
                
                if message_tokens > MAX_TOKENS:
                    # Truncate system message if needed
                    excess_tokens = message_tokens - MAX_TOKENS + 500  # Buffer
                    max_system_tokens = len(encoder.encode(system_message)) - excess_tokens
                    if max_system_tokens < 500:
                        max_system_tokens = 500
                    
                    system_message = truncate_system_message(system_message, max_system_tokens)
                    messages[0]["content"] = system_message
                
                # Call the API with token-optimized messages
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                
                reply_text = response.choices[0].message.content
                
                # For greetings, ensure emoji is present
                if user_greeting:
                    reply_text = add_greeting_emoji(reply_text, True)
                
                # Remove redundant greeting phrases
                reply_text = re.sub(r'^(hello there|hey there|hi there)[,.!]?\s+', '', reply_text, flags=re.IGNORECASE).strip()
                
            except Exception as e:
                print(f"[ERROR] Regular chat error: {str(e)}")
                reply_text = "Sorry, I couldn't process that message. Could you try again?"

        # Save chat history
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

        # Create Twilio response
        twilio_response = MessagingResponse()
        twilio_response.message(reply_text)
        return Response(content=str(twilio_response), media_type="application/xml")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        traceback.print_exc()
        twilio_response = MessagingResponse()
        twilio_response.message("Oops! Something went wrong on our end. Mind trying again?")
        return Response(content=str(twilio_response), media_type="application/xml")