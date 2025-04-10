# eco_chatbotHere's a simple `README.md` for your FastAPI OpenAI chatbot project:

```markdown
# FastAPI OpenAI Chatbot

A simple FastAPI backend that interacts with the OpenAI API to provide chatbot responses.

## Features
- FastAPI endpoint for chatbot interactions
- OpenAI GPT-4o integration
- Request/response validation with Pydantic
- Environment variable configuration
- CORS support

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd project
   ```

2. Install dependencies:
   ```bash
   pip install fastapi uvicorn openai python-dotenv pydantic
   ```

3. Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o
   ```

## Running the Application

Start the server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Chat Endpoint
- **POST** `/chat`
- Request format:
  ```json
  {
      "user_id": "unique_user_id",
      "query": "your message here"
  }
  ```
- Response format:
  ```json
  {
      "user_id": "unique_user_id",
      "response": "chatbot's reply"
  }
  ```

### Health Check
- **GET** `/health`
- Returns:
  ```json
  {"status": "healthy"}
  ```

## Project Structure
```
project/
├── .env                    # Environment variables
├── src/
│   └── constant.py         # System prompts
├── utils/
│   └── utils.py            # OpenAI client functions
├── schema.py               # Pydantic models
└── main.py                # FastAPI application
```

## Requirements
- Python 3.11+
- OpenAI API key
```

You can copy this text and save it as `README.md` in your project root directory. Adjust any details as needed for your specific implementation.