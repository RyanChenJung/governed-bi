"""Test if we can get reasoning trace from OpenAI"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

print("=== Testing OpenAI Reasoning Content ===\n")

# Test with reasoning model
response = client.chat.completions.create(
    model="gpt-5.6-sol",
    messages=[
        {"role": "user", "content": "Calculate 157 * 239. Show your thinking."}
    ],
    reasoning={
        "effort": "medium"  # Try medium to see if we get more reasoning
    }
)

message = response.choices[0].message

print("Message content type:", type(message.content))
print("\nMessage content:")
print(message.content)

print("\n=== Checking for reasoning in response ===")
if hasattr(message, 'reasoning_content'):
    print("✓ Has reasoning_content attribute")
    print(message.reasoning_content)
else:
    print("✗ No reasoning_content attribute")

# Check the raw response
print("\n=== Raw message dict ===")
print(message.model_dump())

print("\n=== Usage ===")
print(response.usage)
