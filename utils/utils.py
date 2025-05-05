import openai
import os
import re
import json
import uuid
import random
import datetime
import traceback
from typing import List, Optional
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from openai import OpenAI
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request as GoogleAuthRequest





# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # Default if not set in .env

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Google Calendar setup
google_credentials_path = 'credentials.json'
google_token_path = 'token.json'
google_scopes = ['https://www.googleapis.com/auth/calendar.events']

# List of WhatsApp-compatible emojis
GREETING_EMOJIS = ["👋", "😊", "👍", "✨", "🌟", "🙂", "👏", "🤗"]

# Rate limiting variables
RATE_LIMITED = False
RATE_LIMIT_UNTIL = datetime.datetime.now()

class EventRequest(BaseModel):
    summary: str
    start_datetime: str
    end_datetime: str
    description: Optional[str] = None
    attendees: Optional[List[EmailStr]] = None

def check_for_greeting(text: str) -> bool:
    """Check if the user's message contains a greeting."""
    greeting_pattern = re.compile(r'^(hello|hi|hey|howdy|hola|yo)(\s+there)?[,.!]?\s*', re.IGNORECASE)
    return bool(greeting_pattern.search(text.strip()))

def check_for_proper_greeting(text: str) -> bool:
    """Check if the response already contains a proper greeting with emoji."""
    # Look for patterns like "Hello👋" or "Hi there😊"
    proper_greeting_pattern = re.compile(r'^(hello|hi|hey|greetings)(\s+there)?[^\w\s]', re.IGNORECASE)
    return bool(proper_greeting_pattern.search(text.strip()))

def remove_greeting(text: str) -> str:
    """Remove any existing greeting from the text."""
    return re.sub(r'^(hello|hi|hey|greetings)(\s+there)?[,.!]?\s+', '', text, flags=re.IGNORECASE)

def get_fallback_response(fallback_type: str) -> str:
    """Get an appropriate fallback response based on the context type."""
    if fallback_type == "greeting":
        emoji = random.choice(GREETING_EMOJIS)
        return f"Hey{emoji} How can I help you today?"
    elif fallback_type == "scheduling":
        return "I'd love to help schedule your appointment. Could you let me know what day and time works best for you? Also, what's your email so I can send a confirmation?"
    elif fallback_type == "service_inquiry":
        return "Thanks for asking about our services! We offer a range of options tailored to different needs. Could you tell me a bit about what you're looking for, and I can give you more specific info?"
    else:
        emoji = random.choice(GREETING_EMOJIS)
        return f"Thanks for reaching out! I got your message but our system is briefly updating. What can I help you with today?"




def get_chat_completion(user_query: str, system_message: str, fallback_type="general") -> str:
    """Generate a chat completion with OpenAI API and handle errors gracefully."""
    global RATE_LIMITED, RATE_LIMIT_UNTIL
    
    # Check if we're currently rate limited
    if RATE_LIMITED and datetime.datetime.now() < RATE_LIMIT_UNTIL:
        print(f"[INFO] Still rate limited until {RATE_LIMIT_UNTIL}. Using fallback.")
        return get_fallback_response(fallback_type)
    
    # Reset rate limit flag if we've passed the time
    if RATE_LIMITED and datetime.datetime.now() >= RATE_LIMIT_UNTIL:
        print("[INFO] Rate limit period ended. Resuming normal operation.")
        RATE_LIMITED = False
    
    # Check if the user's message contains a greeting
    has_greeting = check_for_greeting(user_query)
    
    # IMPORTANT: Handle greeting with emoji in a controlled way
    if has_greeting:
        # If user sent a greeting, tell OpenAI to respond with a greeting and emoji
        enhanced_system_message = system_message + "\n\nIMPORTANT: Begin your response with a friendly greeting like 'Hello' or 'Hi there' and include an emoji right after the greeting word (with no space). For example: 'Hello👋' or 'Hi there😊'. The rest of your response should NOT contain any emojis."
    else:
        # If no greeting detected, proceed with regular response (no greeting or emoji)
        enhanced_system_message = system_message + "\n\nIMPORTANT: DO NOT begin your message with greeting phrases like 'hello there', 'hi there', etc. DO NOT use emojis in your response."
    
    try:
        # Create a completion using the OpenAI API
        print(f"Querying OpenAI with: {user_query}")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": enhanced_system_message},
                {"role": "user", "content": user_query}
            ]
        )
        print(f"OpenAI Response Status: Success")  # Log success without full response details

        # Get the response content
        response_text = completion.choices[0].message.content
        
        # Ensure proper greeting format if needed
        if has_greeting and not check_for_proper_greeting(response_text):
            # If OpenAI didn't provide a proper greeting with emoji, add our own
            emoji = random.choice(GREETING_EMOJIS)
            greeting_options = [
                f"Hello{emoji} ",
                f"Hi there{emoji} ",
                f"Hey{emoji} "
            ]
            # Get just the response content without any existing greeting
            clean_response = remove_greeting(response_text)
            response_text = random.choice(greeting_options) + clean_response
        
        return response_text

    except Exception as e:
        error_message = str(e)
        print(f"Error during OpenAI API call: {error_message}")
        
        # Check if this is a rate limit or quota error
        if "429" in error_message and "insufficient_quota" in error_message:
            # Set rate limit for 15 minutes to reduce repeated errors
            RATE_LIMITED = True
            RATE_LIMIT_UNTIL = datetime.datetime.now() + datetime.timedelta(minutes=15)
            print(f"[RATE LIMIT] API quota exceeded. Rate limited until: {RATE_LIMIT_UNTIL}")
        
        return get_fallback_response(fallback_type)

def get_calendar_service():
    """Get an authenticated Google Calendar service object."""
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

def create_calendar_event(event: EventRequest):
    """Create a Google Calendar event with Meet link."""
    try:
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
                "meet_link": created_event.get('conferenceData', {}).get('entryPoints', [{}])[0].get('uri')
            }
        except HttpError as error:
            print(f"[ERROR] Google Calendar API error: {error}")
            return {"status": "error", "message": f"Calendar error: {str(error)}"}
            
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"[ERROR] Calendar event creation failed: {str(e)}")
        print(f"[ERROR] Traceback: {error_traceback}")
        return {"status": "error", "message": "Failed to create calendar event. Please try again later."}

async def detect_scheduling_intent(query: str) -> bool:
    """Detect if a message is intended for scheduling an appointment."""
    global RATE_LIMITED, RATE_LIMIT_UNTIL
    
    # Check if we're currently rate limited
    if RATE_LIMITED and datetime.datetime.now() < RATE_LIMIT_UNTIL:
        print(f"[INFO] Intent detection: Rate limited until {RATE_LIMIT_UNTIL}. Using fallback.")
        # Simple keyword-based fallback
        scheduling_keywords = [
            "schedule", "appointment", "book", "reserve", "meet", 
            "when can", "available", "calendar", "time slot"
        ]
        return any(keyword in query.lower() for keyword in scheduling_keywords)
    
    try:
        response = client.chat.completions.create(
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
    except Exception as e:
        error_message = str(e)
        print(f"[ERROR] Intent detection error: {error_message}")
        
        # Check if this is a rate limit or quota error
        if "429" in error_message and "insufficient_quota" in error_message:
            # Set rate limit for 15 minutes
            RATE_LIMITED = True
            RATE_LIMIT_UNTIL = datetime.datetime.now() + datetime.timedelta(minutes=15)
            print(f"[RATE LIMIT] API quota exceeded. Rate limited until: {RATE_LIMIT_UNTIL}")
        
        # Fallback to keyword-based detection
        scheduling_keywords = [
            "schedule", "appointment", "book", "reserve", "meet", 
            "when can", "available", "calendar", "time slot"
        ]
        return any(keyword in query.lower() for keyword in scheduling_keywords)
























# sheraz code

# import os
# import numpy as np
# from openai import OpenAI
# from typing import List, Dict

# # Initialize OpenAI client for embeddings
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Maximum words per chunk to respect model context limits
# MAX_WORDS_PER_CHUNK = 200


# def chunk_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
#     """Split text into chunks of up to max_words."""
#     words = text.split()
#     return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]


# def get_relevant_service_chunks(
#     service_docs: List[Dict],
#     user_query: str,
#     max_chunks: int = 3,
#     embed_model: str = "text-embedding-ada-002"
# ) -> List[str]:
#     """
#     Given a list of service documents and a user query, return the top
#     `max_chunks` service description chunks most relevant to the query using
#     embedding-based similarity, with chunking to avoid context overflows.

#     service_docs: List of dicts with keys 'serviceName' and 'description'.
#     user_query: the user's input text to match against.
#     max_chunks: number of top chunks to return.
#     embed_model: embedding model to use.

#     Returns:
#         List of formatted strings: "ServiceName: chunk_text"
#     """
#     # Build all text chunks for all services
#     all_chunks = []
#     for s in service_docs:
#         name = s.get('serviceName', 'Service')
#         desc = s.get('description', '')
#         for chunk in chunk_text(desc):
#             # prefix each chunk with service name for context
#             all_chunks.append(f"{name}: {chunk}")

#     # Compute embeddings for each chunk
#     resp = client.embeddings.create(model=embed_model, input=all_chunks)
#     chunk_embs = [d.embedding for d in resp.data]

#     # Compute embedding for user query
#     q_resp = client.embeddings.create(model=embed_model, input=[user_query])
#     query_emb = q_resp.data[0].embedding

#     # Cosine similarity
#     def cosine_similarity(a, b):
#         a_arr = np.array(a)
#         b_arr = np.array(b)
#         return float(a_arr.dot(b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

#     # Score chunks
#     scores = [cosine_similarity(query_emb, emb) for emb in chunk_embs]

#     # Select top chunk indices
#     top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_chunks]

#     # Return the corresponding text chunks
#     return [all_chunks[i] for i in top_idxs]
