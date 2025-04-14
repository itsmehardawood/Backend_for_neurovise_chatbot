from fastapi import APIRouter, Request, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from utils.utils import get_chat_completion
import os
from dotenv import load_dotenv

load_dotenv()


twilio_router = APIRouter()

@twilio_router.post("/twilio/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...)
):
    # Generate chatbot response
    reply = get_chat_completion(Body)

    # Prepare Twilio response
    twilio_response = MessagingResponse()
    twilio_response.message(reply)

    return Response(content=str(twilio_response), media_type="application/xml")
