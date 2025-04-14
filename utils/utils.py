import openai
import os
from dotenv import load_dotenv
from openai import OpenAI

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