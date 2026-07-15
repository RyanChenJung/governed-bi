"""Test OpenAI reasoning API format"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

print("=== Testing Different Reasoning Formats ===\n")

# Format 1: Try as model parameter
try:
    print("Test 1: Using modalities parameter...")
    response = client.chat.completions.create(
        model="gpt-5.6-sol",
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        store=True,  # Enable storage
        metadata={"reasoning_effort": "low"}
    )
    
    message = response.choices[0].message
    print(f"Success! Content: {message.content[:100]}")
    
    # Check if there's thinking in the response
    if hasattr(message, '__dict__'):
        print("\nMessage attributes:", list(message.__dict__.keys()))
    
    print("\nFull message model_dump:")
    import json
    print(json.dumps(message.model_dump(), indent=2))
    
except Exception as e:
    print(f"Failed: {e}")
