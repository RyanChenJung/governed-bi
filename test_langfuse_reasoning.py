"""Test if Langfuse captures reasoning metadata"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from governed_bi.obs import tracing_callbacks

load_dotenv()

print("=== Testing Langfuse with Reasoning Model ===\n")

model = ChatOpenAI(
    model="gpt-5.6-sol",
    model_kwargs={"reasoning": {"effort": "medium"}}
)

callbacks = tracing_callbacks()
print(f"Callbacks: {len(callbacks)}")

response = model.invoke(
    "Calculate 23 * 47 step by step",
    config={"callbacks": callbacks}
)

print("\n=== Response Content Blocks ===")
if isinstance(response.content, list):
    for i, block in enumerate(response.content):
        print(f"\nBlock {i}: {block.get('type')}")
        if block.get('type') == 'reasoning':
            print(f"  ID: {block.get('id', '')[:40]}...")
            print(f"  Has encrypted_content: {bool(block.get('encrypted_content'))}")
            print(f"  Encrypted length: {len(block.get('encrypted_content', ''))}")
            print(f"  Summary: {block.get('summary', [])}")
        elif block.get('type') == 'text':
            print(f"  Text: {block.get('text', '')[:100]}")

print("\n✓ Done! Check Langfuse for this trace.")
print("  The reasoning metadata should be visible even if content is encrypted.")
