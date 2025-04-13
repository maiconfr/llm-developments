import os
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

year = ['2020', '1964']

template_str = "What happened in {country} in {year}"
prompt = PromptTemplate(template=template_str)

formatted_prompt = prompt.format(country="Brazil", year=year[1])
print(formatted_prompt)

llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)

response = llm.complete(formatted_prompt)
print(response.text)
