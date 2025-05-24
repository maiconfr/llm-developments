from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from typing_extensions import List, TypedDict
# from langchain import hub
# from langchain_core.documents import Document

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

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)


def create_documents(folder_path):
        documents = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith('.pdf'):            
                loader = PyPDFLoader(file_path)
            
            if file.endswith('.txt'):
                loader = TextLoader(file_path)

            documents.extend(loader.load())
        
        return documents


documents = []
documents.extend(create_documents(folder_path="files"))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
chunks = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

query_text = "Explain how LLM works in simple terms."


def query_rag(query_text):
  # Retrieving the context from the DB using similarity search
  results = vectorstore.similarity_search_with_relevance_scores(query_text, k=3)

  # Check if there are any matching results or if the relevance score is too low
  if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")

  context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
 
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)
  
  model = ChatOpenAI()

  # Generate response text based on the prompt
  response_text = model.predict(prompt)
 
   # Get sources of the matching documents
  sources = [doc.metadata.get("source", None) for doc, _score in results]
 
  # Format and return response including generated text and sources
  formatted_response = f"Response: {response_text}\nSources: {sources}"
  return formatted_response, response_text

formatted_response, response_text = query_rag(query_text)

print(response_text)