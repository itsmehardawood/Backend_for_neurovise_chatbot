# import openai
# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()  # Load environment variables from .env file

# openai.api_key = os.getenv("OPENAI_API_KEY")
# model = os.getenv("OPENAI_MODEL")  # Make sure OPENAI_MODEL is set in your .env
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def get_chat_completion(user_query: str, system_message: str) -> str:
    
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
    
    
# #----------------------------------------------------
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
#             "meet_link": created_event.get('conferenceData', {}).get('entryPoints', [{}])[0].get('uri')
#         }
#     except HttpError as error:
#         return {"status": "error", "message": str(error)}








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




import openai
import os
import re
import random
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")  # Make sure OPENAI_MODEL is set in your .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List of WhatsApp-compatible emojis
GREETING_EMOJIS = ["👋", "😊", "👍", "✨", "🌟", "🙂", "👏", "🤗"]

def get_chat_completion(user_query: str, system_message: str) -> str:
    # Check if the user's message contains a greeting
    has_greeting = check_for_greeting(user_query)
    
    # IMPORTANT CHANGE: Instead of telling OpenAI to NOT use emojis,
    # we're now telling it to use them but in a controlled way
    if has_greeting:
        # If user sent a greeting, tell OpenAI to respond with a greeting and emoji
        enhanced_system_message = system_message + "\n\nIMPORTANT: Begin your response with a friendly greeting like 'Hello' or 'Hi there' and include an emoji right after the greeting word (with no space). For example: 'Hello👋' or 'Hi there😊'. The rest of your response should NOT contain any emojis."
    else:
        # If no greeting detected, proceed with regular response (no greeting or emoji)
        enhanced_system_message = system_message + "\n\nIMPORTANT: DO NOT begin your message with greeting phrases like 'hello there', 'hi there', etc. DO NOT use emojis in your response."
    
    try:
        # Create a completion using the OpenAI API
        print(f"Querying OpenAI with: {user_query}")
        completion = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": enhanced_system_message},
                {"role": "user", "content": user_query}
            ]
        )
        print(f"OpenAI Completion Response: {completion}")  # Log the response

        # Get the response content
        response_text = completion.choices[0].message.content
        
        # IMPORTANT: Instead of post-processing to remove emojis, we'll now
        # just make sure the format is consistent with what we want
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
        # Handle any potential errors
        print(f"Error during OpenAI API call: {e}")
        emoji = random.choice(GREETING_EMOJIS)
        return f"Hello{emoji} Sorry, there was an error processing your request."


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


# The rest of your calendar code remains unchanged
# ----------------------------------------------------
# ... [rest of the calendar code remains the same]




# The rest of your calendar code remains unchanged
# ----------------------------------------------------
# ... [rest of the calendar code remains the same]

    
#----------------------------------------------------
import os
import json
import uuid
import traceback
from datetime import datetime
from typing import List, Optional
from fastapi import HTTPException, Request
from bson import ObjectId
from pydantic import BaseModel, EmailStr
from openai import OpenAI
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from fastapi import Request as FastAPIRequest
from google.auth.transport.requests import Request as GoogleAuthRequest


# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
google_credentials_path = 'credentials.json'
google_token_path = 'token.json'
google_scopes = ['https://www.googleapis.com/auth/calendar.events']

class EventRequest(BaseModel):
    summary: str
    start_datetime: str
    end_datetime: str
    description: Optional[str] = None
    attendees: Optional[List[EmailStr]] = None

def get_calendar_service():
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
        return {"status": "error", "message": str(error)}

async def detect_scheduling_intent(query: str) -> bool:
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
