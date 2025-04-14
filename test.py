import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()  # Load .env variables

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

message = client.messages.create(
    from_="whatsapp:+14155238886",  # Sandbox number
    to="whatsapp:+923067145010",  # Your verified WhatsApp number
    body="Hello from Eco Chatbot!"
)

print(message.sid)
