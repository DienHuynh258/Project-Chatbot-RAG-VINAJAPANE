# src/chatbot/core/utils.py
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from config import GEMINI_MODEL_NAME, LLM_TEMPERATURE, VECTOR_STORE_DIR, RETRIEVER_SEARCH_K
import sys
from pathlib import Path
from langchain_community.vectorstores import Chroma


def get_embedding_model():
    """
    Tải và trả về model embedding local (MiniLM).
    Hàm này được dùng chung bởi cả ingest.py và chain.py
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'} # Ép chạy trên CPU
    encode_kwargs = {'normalize_embeddings': True} # Quan trọng
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    

def get_llm():
    """
    Khởi tạo và trả về LLM (Gemini).
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            google_api_key=os.environ.get("GOOGLE_API_KEY") # Tự động đọc từ .env
        )
        print(f"LOG: Đã tải model Gemini ({GEMINI_MODEL_NAME}).")
        return llm
    except Exception as e:
        print(f"LỖI: Không thể tải Gemini. Bạn đã set GOOGLE_API_KEY trong file .env chưa? Lỗi: {e}")
        return None
    

def get_retriever():
    """
    Tải Vector Store và tạo một retriever.
    """
    
    # 1. Tải embedding model
    print("LOG: Đang tải model embedding (MiniLM)...")
    embeddings = get_embedding_model()
    
    if not VECTOR_STORE_DIR.exists():
        print(f"LỖI: Thư mục Vector Store không tồn tại: {VECTOR_STORE_DIR}")
        raise FileNotFoundError(f"Thư mục Vector Store không tồn tại: {VECTOR_STORE_DIR}")
        
    # 2. Tải vector store
    print(f"LOG: Đang tải Vector Store từ: {VECTOR_STORE_DIR}")
    vector_store = Chroma(
        persist_directory=str(VECTOR_STORE_DIR), 
        embedding_function=embeddings
    )
    
    # 3. Tạo retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': RETRIEVER_SEARCH_K} 
    )
    print(f"LOG: Đã tạo retriever (k={RETRIEVER_SEARCH_K}) từ Vector Store.")
    return retriever
