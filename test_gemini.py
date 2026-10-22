
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Explain supplier risk in supply chain management in two sentences."
)

print(response.text)
