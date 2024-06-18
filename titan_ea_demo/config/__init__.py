import os
import ollama
import chromadb

from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

LOCAL_LLM = "llama3"

OLLAMA_HOST = "localhost"
if not os.getenv("OLLAMA_HOST") is None:
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = "11434"
if not os.getenv("OLLAMA_PORT") is None:
    OLLAMA_PORT = os.getenv("OLLAMA_PORT")

print(f"Connecting to Ollama at: {OLLAMA_HOST}:{OLLAMA_PORT}")    
OLLAMA_CLIENT = ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

TEXT_LLM = None
JSON_LLM = None
if OLLAMA_HOST:
    TEXT_LLM = ChatOllama(
        model=LOCAL_LLM, temperature=0, base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/"
    )
    JSON_LLM = ChatOllama(
        model=LOCAL_LLM,
        format="json",
        temperature=0,
        base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/",
    )
else:
    TEXT_LLM = ChatOllama(model=LOCAL_LLM, temperature=0)
    JSON_LLM = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)

CHROMADB_HOST = "localhost"
if not os.getenv("CHROMADB_HOST") is None:
    CHROMADB_HOST = os.getenv("CHROMADB_HOST")
CHROMADB_PORT = "8000"
if not os.getenv("CHROMADB_PORT") is None:
    CHROMADB_PORT = os.getenv("CHROMADB_PORT")

CHROMADB_CLIENT = None

print(f"Connecting to ChromaDB at: {CHROMADB_HOST}:{CHROMADB_PORT}")    
if CHROMADB_HOST:
    CHROMADB_CLIENT = chromadb.HttpClient(host=CHROMADB_HOST, port=int(CHROMADB_PORT))
else:
    CHROMADB_CLIENT = chromadb.HttpClient(host='127.0.0.1', port=8000)
    
API_LISTENER_ADDRESS = "0.0.0.0"
if not os.getenv("API_LISTENER_ADDRESS") is None:
    API_LISTENER_ADDRESS = os.getenv("API_LISTENER_ADDRESS")
API_LISTENER_PORT='7860'
if not os.getenv("API_LISTENER_PORT") is None:
    API_LISTENER_PORT = os.getenv("API_LISTENER_PORT")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name=model_name)

DEFAULT_URLS = [
    "https://clouddocs.f5.com/service-proxy/latest/intro.html",
    "https://clouddocs.f5.com/service-proxy/latest/spk-bgp-overview.html",
    "https://clouddocs.f5.com/service-proxy/latest/spk-network-overview.html",
    "https://clouddocs.f5.com/service-proxy/latest/spk-secure-spk-deployment.html",
    "https://clouddocs.f5.com/service-proxy/latest/spk-tmm-resources.html",
    "https://clouddocs.f5.com/cnfs/robin/latest/intro.html",
    "https://clouddocs.f5.com/cnfs/robin/latest/cnf-firewall-crd.html",
    "https://clouddocs.f5.com/cnfs/robin/latest/cnf-context-global.html",
]

WELCOME_MARKDOWN_FILE = "assets/welcome.md"
SPK_SERVICE_SETUP_FILE = "assets/spk_setup.md"
QUERY_MARKDOWN_FILE = "assets/query_doc.md"
RAG_TRAINING_MARKDOWN_FILE = "assets/rag_training.md"
ABOUT_RAG_MARKDOWN_FILE = "assets/about_rag.md"

__all__ = [
    "LOCAL_LLM",
    "OLLAMA_HOST",
    "OLLAMA_PORT",
    "TEXT_LLM",
    "JSON_LLM",
    "CHROMADB_HOST",
    "CHROMADB_PORT",
    "CHROMADB_CLIENT",
    "API_LISTENER_ADDRESS",
    "API_LISTENER_PORT",
    "EMBEDDINGS_MODEL",
    "DEFAULT_URLS",
    "ABOUT_MARKDOWN_FILE",
]