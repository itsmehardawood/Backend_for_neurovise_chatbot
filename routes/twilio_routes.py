
 
 
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
            "content": "Determine if user want to an appointment or want to schedule a meeting related to it. Reply only 'yes' or 'no'."
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

For appointment scheduling, response should be this :

Can you provide the following details?
1. Meeting topic
2. Start date and time
3. End date and time
4. email address


if user asking appointment in hebrew then response this :
    
האם תוכל לספק את הפרטים הבאים?
1. נושא הפגישה
2. תאריך ושעת התחלה
3. תאריך ושעה סיום
4. כתובת אימייל

Keep it conversational and friend and to the point directly - like texting a colleague."""
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
            f"Service: {s['serviceName']}\nDescription: {s['description']}...\nPrice: {s.get('price', 'N/A')}"
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
                print("here is the is scheduling : ", is_scheduling)
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
                                "summary": {"type": "string", "description": "Meeting Topic"},
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
