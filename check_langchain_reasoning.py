"""Check how LangChain handles reasoning models"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json

load_dotenv()

print("=== Testing LangChain with Reasoning Model ===\n")

# Create model with reasoning
model = ChatOpenAI(
    model="gpt-5.6-sol",
    model_kwargs={
        "reasoning": {"effort": "medium"}
    }
)

print("Model config:")
print(f"  Model: {model.model_name}")
print(f"  Model kwargs: {model.model_kwargs}")

# Invoke with a query that benefits from reasoning
response = model.invoke("Calculate 157 * 239. Think step by step.")

print("\n=== Response ===")
print(f"Type: {type(response)}")
print(f"Content type: {type(response.content)}")
print(f"\nContent:\n{response.content}")

print("\n=== Response Metadata ===")
if hasattr(response, 'response_metadata'):
    print(json.dumps(response.response_metadata, indent=2))

print("\n=== Additional Kwargs ===")
if hasattr(response, 'additional_kwargs'):
    print(json.dumps(response.additional_kwargs, indent=2))

print("\n=== Checking for reasoning content ===")
# Check if reasoning is in content blocks
if isinstance(response.content, list):
    print("Content is a list of blocks:")
    for i, block in enumerate(response.content):
        print(f"\nBlock {i}:")
        print(f"  Type: {block.get('type', 'unknown')}")
        if block.get('type') == 'thinking':
            print(f"  ✓ FOUND THINKING BLOCK!")
            print(f"  Thinking: {block.get('thinking', '')[:200]}...")
        elif block.get('type') == 'text':
            print(f"  Text: {block.get('text', '')[:100]}...")
else:
    print(f"Content is a string: {str(response.content)[:200]}")
