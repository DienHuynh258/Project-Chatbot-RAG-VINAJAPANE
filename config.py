# config.py
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DOCS_DIR = PROJECT_ROOT / "data" / "source_docs" 
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store" / "global"

# --- THÊM DÒNG NÀY ---
JSON_OUTPUT_DIR = PROJECT_ROOT / "data" / "json_output" 
# ---------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- CẤU HÌNH RAG ---
# 1. LLM
# Lấy tên model từ biến môi trường, nếu không có thì dùng "gemini-2.5-flash"
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
LLM_TEMPERATURE = 0.3

# 2. Retriever
# Số lượng 'k' tài liệu sẽ lấy
RETRIEVER_SEARCH_K = 3