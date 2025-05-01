import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
from constant import system_message

load_dotenv()  # Load environment variables from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")  # Make sure OPENAI_MODEL is set in your .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_chat_completion(user_query: str, system_message: str) -> str:
    try:
        # Create a completion using the OpenAI API
        print(f"Querying OpenAI with: {user_query}")
        completion = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query}
            ]
        )
        print(f"OpenAI Completion Response: {completion}")  # Log the response

        # Return the content of the chat response
        return completion.choices[0].message.content

    except Exception as e:
        # Handle any potential errors, such as connection issues or invalid responses
        print(f"Error during OpenAI API call: {e}")
        return "Sorry, there was an error processing your request."
    
    
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
