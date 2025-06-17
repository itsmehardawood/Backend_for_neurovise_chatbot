# import os
# from dotenv import load_dotenv
# from twilio.rest import Client

# load_dotenv()  # Load .env variables

# account_sid = os.environ["TWILIO_ACCOUNT_SID"]
# auth_token = os.environ["TWILIO_AUTH_TOKEN"]
# client = Client(account_sid, auth_token)

# message = client.messages.create(
#     from_="whatsapp:+14155238886",  # Sandbox number
#     to="whatsapp:+923067145010",  # Your verified WhatsApp number
#     body="Hello from Eco Chatbot!"
# )

# print(message.sid)



from openai import OpenAI
client = OpenAI(api_key="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")

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn."
        }
    ]
)

print(completion.choices[0].message.content)