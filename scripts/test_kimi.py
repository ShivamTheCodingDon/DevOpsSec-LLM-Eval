import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

def test_single_call():
    load_dotenv()
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    if not api_key:
        print("API Key not found!")
        return

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    
    try:
        response = client.chat.completions.create(
            model="Kimi-K2.7-Code", # Azure deployment name
            messages=[{"role": "user", "content": "Hello, are you Kimi-K2.7-Code?"}],
            max_tokens=50,
            temperature=0.0
        )
        print("Success!")
        print(response.choices[0].message.content)
        if response.usage:
            print("Usage:", response.usage)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_call()
