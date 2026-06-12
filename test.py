"""Quick smoke test for the Gemini API. Requires GEMINI_API_KEY in .env."""

import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = (os.environ.get("GEMINI_API_KEY") or "").strip().strip("\"'")
if not api_key or api_key == "your_gemini_api_key_here":
    print("Error: GEMINI_API_KEY is not set.")
    print("  Edit .env and replace the placeholder with your real key from AI Studio.")
    sys.exit(1)
if not (api_key.startswith("AIza") or api_key.startswith("AQ.")):
    print("Warning: key doesn't start with AIza or AQ. — double-check you copied it correctly.")

model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model=model,
        contents="Say 'ResearchGPT is ready' in exactly those words.",
    )
    print(response.text)
except Exception as e:
    err = str(e)
    if "429" in err or "RESOURCE_EXHAUSTED" in err:
        print(f"Error: quota exceeded for model '{model}'.")
        print("  Your API key works, but this model has no free-tier quota left.")
        print("  Try: set GEMINI_MODEL=gemini-2.5-flash in .env (or wait ~1 min and retry).")
        print("  Check limits: https://aistudio.google.com/rate-limit")
    else:
        print(f"Error: {e}")
    sys.exit(1)
