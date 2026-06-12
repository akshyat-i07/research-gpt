from google import genai

client = genai.Client(api_key="AQ.Ab8RN6IaiA6A2Hv23YiJtrfCe9BeLw-CqHpIB3XSA0PuQ5EvuQ")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello"
)

print(response.text)