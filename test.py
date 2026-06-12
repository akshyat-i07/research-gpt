"""Quick smoke test for Gemini API connectivity. Requires GEMINI_API_KEY in .env or environment."""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    raise SystemExit("Set GEMINI_API_KEY in .env or environment before running test.py")

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
    contents="Say hello in one word.",
)
print(response.text)
