# Quick test to verify Gemini API key is working
from google import genai
import os
import sys

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAXwIjvKv-xQyfmJpNtz1M-zOFW1W9Kdvk")

print("=" * 60)
print("Testing Gemini API Key")
print("=" * 60)
print(f"API Key present: {bool(GEMINI_API_KEY)}")
print(f"API Key length: {len(GEMINI_API_KEY)}")
print(f"API Key starts with 'AIza': {GEMINI_API_KEY.startswith('AIza')}")
print()

if not GEMINI_API_KEY or GEMINI_API_KEY == "your-api-key-here":
    print("[ERROR] Please set a valid API key!")
    print("Get one from: https://aistudio.google.com/apikey")
    print("Then run: $env:GEMINI_API_KEY = 'your-api-key'")
    sys.exit(1)

try:
    print("Creating Gemini client...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("[OK] Client created successfully")
    
    print("\nTesting API call with simple message...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'Hello, API is working!' in one sentence."
    )
    
    print("[OK] API call successful!")
    print("\nResponse:")
    print("-" * 60)
    
    if hasattr(response, "text"):
        print(response.text)
    elif hasattr(response, "candidates") and len(response.candidates) > 0:
        print(response.candidates[0].content.parts[0].text)
    else:
        print(f"Response object: {response}")
        print(f"Response type: {type(response)}")
        print(f"Available attributes: {[a for a in dir(response) if not a.startswith('_')]}")
    
    print("-" * 60)
    print("\n[SUCCESS] Your API key is valid and working!")
    print("You can now run main.py")
    
except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print()
    if "API Key" in str(e) or "API_KEY" in str(e) or "INVALID_ARGUMENT" in str(e):
        print("The API key is invalid or expired.")
        print("Please get a new API key from: https://aistudio.google.com/apikey")
    import traceback
    traceback.print_exc()
    sys.exit(1)

