import time
import re
import titan_ea_demo.config as config

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

from langchain_community.tools.tavily_search import TavilySearchResults

vectorstore = None
vectorstore_retriever = None


def time_stamper(start_time):
    run_duration = round((time.time() - start_time), 3)
    return f"Runtime: ({run_duration} seconds)\n"


def url_splitter(urls):
    return_urls = []
    if isinstance(urls, str):
        return_urls = re.split(r"\s+|,\s+|\n|\t", urls)
    return return_urls


def web_search(query):
    web_search_tool = TavilySearchResults(k=20)
    docs = web_search_tool.invoke({"query": query})
    return docs


def ollama_pull_model():
    loaded_modules = config.OLLAMA_CLIENT.list()
    llama_needed = True
    for model in loaded_modules["models"]:
        if config.LOCAL_LLM in model["name"]:
            return False
    config.OLLAMA_CLIENT.pull(config.LOCAL_LLM)
    return True


def get_vectorstore(text_tokens):
    global vectorstore, vectorstore_retriever
    vectorstore = Chroma.from_documents(
        client=config.CHROMADB_CLIENT,
        documents=text_tokens,
        collection_name="rag-chroma",
        embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
    )
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return vectorstore


def get_vectorstore_retriever():
    return vectorstore_retriever


def clear_rag_training():
    global vectorstore, vectorstore_retriever
    delete_handle = Chroma(
        client=config.CHROMADB_CLIENT,
        collection_name="rag-chroma"
    )
    delete_handle.delete_collection()
    vectorstore = None
    vectorstore_retriever = None
    

__all__ = [
    "time_stamper",
    "url_splitter",
    "web_search",
    "ollama_pull_model",
    "get_vectorstore",
    "get_vectorstore_retriever",
    "clear_rag_training"
]
