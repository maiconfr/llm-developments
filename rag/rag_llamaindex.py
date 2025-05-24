from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import TokenTextSplitter

import os
try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Configure chat engine
chat_engine = SimpleChatEngine.from_defaults(llm=llm, verbose=True)


def create_documents(folder_path):
        reader = SimpleDirectoryReader(input_dir=folder_path)
        documents = reader.load_data()
        return documents

documents = []
documents.extend(create_documents(folder_path="./files"))

text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)

index = VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine(llm=llm, verbose=True)

response = query_engine.query(
    "Explain how LLM works in simple terms."
)

print(str(response))



