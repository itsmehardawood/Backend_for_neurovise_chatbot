from openai import OpenAI
import os
from dotenv import load_dotenv
from src.constant import system_message

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPENAI_MODEL")

def get_chat_completion(user_query: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ]
    )
    return completion.choices[0].message.content