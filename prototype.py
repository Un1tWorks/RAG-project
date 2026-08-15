import os
import sys

# Windows threading fixes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

# Persistent Storage Imports
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# 1. Models & Chunking Setup
Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.text_splitter = SentenceSplitter(chunk_size=300, chunk_overlap=30)

def get_or_create_index(data_dir: str):
    # Setup local ChromaDB folder on hard drive
    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection("raiffeisen_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Check if vectors already exist on disk
    if chroma_collection.count() > 0:
        print("[+] Existing vector database found on disk! Loading instantly...", flush=True)
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=Settings.embed_model
        )
    else:
        print("[+] First time setup: Extracting text & building persistent vector database...", flush=True)
        reader = PyMuPDFReader()
        documents = []
        for file in os.listdir(data_dir):
            if file.endswith(".pdf"):
                file_path = os.path.join(data_dir, file)
                print(f"    - Loading: {file}", flush=True)
                documents.extend(reader.load_data(file_path=file_path))

        print(f"[+] Calculating embeddings for {len(documents)} pages (Only happens ONCE)...", flush=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            show_progress=True
        )
        print("[+] Success! Vectors saved to ./chroma_db", flush=True)

    return index

def run_raiffeisen_rag(data_dir: str, customer_query: str):
    index = get_or_create_index(data_dir)

    print("[+] Querying Raiffeisen Support Engine...", flush=True)
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        system_prompt=(
            "You are a helpful customer service assistant for Raiffeisen Bank. "
            "Answer the customer's question clearly and politely based ONLY on the provided context. "
            "If the information is not in the documents, state that you do not have that detail."
        )
    )
    
    response = query_engine.query(customer_query)
    
    print("\n" + "="*20 + " RETRIEVED BANKING POLICIES " + "="*20, flush=True)
    for i, source_node in enumerate(response.source_nodes):
        score = source_node.score
        page_num = source_node.node.metadata.get("page_label", "Unknown")
        file_name = source_node.node.metadata.get("file_name", "Unknown")
        snippet = source_node.node.get_content()[:200].replace("\n", " ")
        print(f"\n[Match {i+1}] (Score: {score:.4f} | Source: {file_name} | Page: {page_num})", flush=True)
        print(f"Snippet: \"{snippet}...\"", flush=True)
    
    return response

if __name__ == "__main__":
    CUSTOMER_QUESTION = "Ce trebuie să fac dacă am pierdut cardul de debit și vreau să-l blochez?"
    result = run_raiffeisen_rag(data_dir="./data", customer_query=CUSTOMER_QUESTION)
    
    print("\n" + "="*20 + " RAIFFEISEN ASSISTANT ANSWER " + "="*20, flush=True)
    print(result, flush=True)