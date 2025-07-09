
 
 
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

def prepare_whatsapp_system_message(business_settings, user_query):
    """
    Single optimized system message for WhatsApp that handles everything
    """
    base_message = """You are a friendly WhatsApp business assistant. Rules:

LANGUAGE: Default Hebrew (unless user requests English)

AUTO-RESPONSES:
- Greeting (hi/hello/hey) → Start with greeting + emoji: "שלום👋" or "Hello👋"
- Service inquiry → List available services naturally
- Scheduling → Ask for details or use function if provided

SCHEDULING FORMAT:
Hebrew: "האם תוכל לספק את הפרטים הבאים? 1. נושא הפגישה 2. תאריך ושעת התחלה 3. תאריך ושעה סיום 4. כתובת אימייל"
English: "Can you provide: 1. Meeting topic 2. Start date/time 3. End date/time 4. Email address"

STYLE: Conversational, human-like, brief (like texting)"""

    if business_settings:
        if business_settings.get('system_prompt'):
            base_message += f"\n\nBUSINESS: {business_settings['system_prompt']}"
        
        active_services = [s for s in business_settings.get('services', []) if s.get('isActive', True)]
        if active_services:
            services = "\n".join(f"- {s['serviceName']}: {s.get('description', '')[:60]}..." 
                               for s in active_services[:5])
            base_message += f"\n\nSERVICES:\n{services}"
        
        tone = business_settings.get('chat_tone', 'friendly')
        base_message += f"\nTONE: {tone}"
    
    return truncate_system_message(base_message, 2500)

@twilio_router.post("/twilio/whatsapp")
async def whatsapp_webhook_optimized(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
):
    """
    Optimized WhatsApp handler - single OpenAI call instead of multiple
    """
    try:
        # Extract numbers
        customer_number = From.replace("whatsapp:", "").replace("+", "").strip()
        owner_number = To.replace("whatsapp:", "").replace("+", "").strip()

        # Find owner
        owner_user = users_collection.find_one({"phone_number": owner_number})
        if not owner_user:
            return Response(status_code=200)

        owner_user_id = str(owner_user["_id"])
        business_settings = business_settings_collection.find_one({"user_id": ObjectId(owner_user_id)})
        
        # Prepare single optimized system message
        system_message = prepare_whatsapp_system_message(business_settings, Body)
        
        # Simple keyword detection (no AI needed)
        is_scheduling = has_scheduling_keywords(Body)
        
        # Create messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": Body}
        ]
        
        # Create API parameters
        api_params = {
            "model": "gpt-3.5-turbo",
            "messages": messages
        }
        
        # Add tools only if scheduling detected
        if is_scheduling:
            api_params["tools"] = [{
                "type": "function",
                "function": {
                    "name": "create_calendar_event",
                    "description": "Schedule appointment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "start_datetime": {"type": "string"},
                            "end_datetime": {"type": "string"},
                            "description": {"type": "string"},
                            "attendees": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["summary", "start_datetime", "end_datetime"]
                    }
                }
            }]
            api_params["tool_choice"] = "auto"
        
        # Single API call
        response = openai_client.chat.completions.create(**api_params)

        message = response.choices[0].message
        
        # Handle scheduling tool calls
        if message.tool_calls:
            try:
                event_data = json.loads(message.tool_calls[0].function.arguments)
                event_data['summary'] = event_data.get('summary') or "Appointment"
                
                if owner_user.get('email'):
                    event_data.setdefault('attendees', []).append(owner_user['email'])

                event_result = create_calendar_event(EventRequest(**event_data))

                if event_result["status"] == "success":
                    reply_text = f"✅ נקבע! {event_data['summary']} ב-{event_data['start_datetime'].split('T')[0]}"
                    if event_result.get('meet_link'):
                        reply_text += f"\n🔗 {event_result['meet_link']}"
                else:
                    reply_text = f"❌ שגיאה בקביעת התור: {event_result.get('message', 'נסה שוב')}"
            except:
                reply_text = "צריך עוד פרטים לקביעת התור. תאריך, שעה ונושא?"
        else:
            reply_text = message.content

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

        # Return response
        twilio_response = MessagingResponse()
        twilio_response.message(reply_text)
        return Response(content=str(twilio_response), media_type="application/xml")

    except Exception as e:
        print(f"WhatsApp error: {str(e)}")
        twilio_response = MessagingResponse()
        twilio_response.message("שגיאה טכנית, נסה שוב")
        return Response(content=str(twilio_response), media_type="application/xml")