from langchain_openai import ChatOpenAI

import os
try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set debug to see what is going on with the LLM
# This is useful for debugging and understanding how the LLM works
from langchain.globals import set_debug
set_debug(True)

country = "Brazil"

prompt = f"Tell me about {country}."

print(prompt)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)

response = llm.invoke(prompt)
print(response.content)
