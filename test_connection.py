# test_connection.py — ganti isinya
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Model chat (generateContent) yang tersedia:\n")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  ✅ {m.name}")