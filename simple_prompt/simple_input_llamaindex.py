from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine

import os
try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


country = "Brazil"

prompt = f"Tell me about {country}."

print(prompt)

llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Configure chat engine
chat_engine = SimpleChatEngine.from_defaults(llm=llm, verbose=True)

response = chat_engine.chat(prompt)
print(response)
