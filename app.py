import os
import sys
import tempfile
import nltk

# Configure NLTK to use a writable temporary directory for Streamlit Cloud
nltk_data_dir = os.path.join(tempfile.gettempdir(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
    except Exception:
        pass

# Prevent multithreading locks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# Support API key from Streamlit Cloud Secrets or local .env
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Raiffeisen Bank Assistant",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Raiffeisen Customer Support AI")
st.caption("Ask any question regarding banking conditions, fees, and account policies.")

@st.cache_resource(show_spinner="Initializing AI Engine and Vector DB...")
def load_rag_engine():
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection("raiffeisen_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=Settings.embed_model
    )
    
    return index.as_query_engine(
        similarity_top_k=3,
        system_prompt=(
            "You are a helpful, polite customer service assistant for Raiffeisen Bank. "
            "Answer the customer's question clearly and concisely based ONLY on the provided context. "
            "If the answer is not in the documents, state that you do not have that detail."
        )
    )

query_engine = load_rag_engine()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bună ziua! How can I help you with Raiffeisen banking services today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ex: Care sunt condițiile generale pentru persoane fizice?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching bank documents..."):
            response = query_engine.query(user_query)
            st.markdown(str(response))
            
            with st.expander("📄 View Source Citations"):
                for i, node in enumerate(response.source_nodes):
                    file_name = node.node.metadata.get("file_name", "Document")
                    page_num = node.node.metadata.get("page_label", "N/A")
                    score = node.score
                    snippet = node.node.get_content()[:200].replace("\n", " ")
                    st.write(f"**[{i+1}] {file_name} (Page {page_num})** — *Match Score: {score:.4f}*")
                    st.caption(f'"{snippet}..."')

    st.session_state.messages.append({"role": "assistant", "content": str(response)})