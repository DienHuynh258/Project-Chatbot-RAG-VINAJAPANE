# run_ingest.py
# (Giai đoạn 2: Đọc JSON -> Chunk -> Embed -> Nạp vào Vector Store)

import os
import sys
import json
import re
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import JSON_OUTPUT_DIR, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.chatbot.core.utils import get_embedding_model

# --- LOGIC CHUNKING (Chuyển từ file cũ sang) ---

def chunk_unstructured_elements(elements: list[dict]) -> list[Document]:
    """
    Chunk 1 list các 'elements' (dạng dict) từ file JSON.
    Đây là logic 'chunk_elements' cũ của anh, nhưng đọc từ dict.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    current_text_batch = ""

    for el in elements:
        el_type = el.get("type", "").lower()
        text = el.get("text", "")
        
        if el_type == "table":
            if current_text_batch.strip():
                chunks.extend(text_splitter.create_documents([current_text_batch]))
                current_text_batch = ""
            
            # Xử lý bảng
            table_text = el.get("text")
            chunks.append(Document(page_content=f"Đây là một bảng: {table_text}", metadata={"type": "table"}))
        
        elif el_type in ("narrativetext", "listitem", "title"):
            current_text_batch += text + "\n\n"
    
    if current_text_batch.strip():
        chunks.extend(text_splitter.create_documents([current_text_batch]))
    
    return chunks

def chunk_table_rows(rows: list[dict]) -> list[Document]:
    """
    Chunk dữ liệu dạng hàng (từ CSV/DAT). Mỗi hàng là 1 Document.
    Đây là logic 'process_csv_file' cũ của anh.
    """
    chunks = []
    for i, row_dict in enumerate(rows):
        content_parts = [f"{str(col).strip()}: {str(val).strip()}" for col, val in row_dict.items()]
        page_content = ", ".join(content_parts)
        metadata = {"type": "csv_row", "row_index": i + 1}
        chunks.append(Document(page_content=page_content, metadata=metadata))
    return chunks

def chunk_plain_text(content: str) -> list[Document]:
    """Chunk file text đơn giản (từ .txt)"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.create_documents([content])

def chunk_code(content: dict) -> list[Document]:
    """Chunk file code (logic từ 'process_code_file' cũ)"""
    lang = content.get("language", "text")
    text = content.get("content", "")
    
    try:
        lang_enum = Language(lang) # Cố gắng map sang enum
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    except: # Nếu thất bại (ví dụ lang="text"), dùng splitter mặc định
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        
    return text_splitter.create_documents([text])

# --- HÀM MAIN CỦA GIAI ĐOẠN 2 ---

def main():
    print("--- BẮT ĐẦU GIAI ĐOẠN 2: INGEST JSON VÀO VECTOR STORE ---")
    
    if not JSON_OUTPUT_DIR.exists():
        print(f"Thư mục JSON {JSON_OUTPUT_DIR} không tồn tại. Hãy chạy run_parser.py trước.")
        return

    print("Đang tải model embedding...")
    embeddings = get_embedding_model()
    
    print("Khởi tạo Chroma Vector Store...")
    store_path = VECTOR_STORE_DIR / "global"
    store_path.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        persist_directory=str(store_path),
        embedding_function=embeddings
    )

    # Lấy metadata cũ để check trùng lặp
    existing_metadatas = []
    try:
        existing_metadatas = vectorstore.get()["metadatas"]
    except Exception:
        pass

    def is_duplicate(topic, doc_id):
        return any(
            meta.get("topic") == topic and meta.get("document_id") == doc_id
            for meta in existing_metadatas
        )

    print(f"Quét thư mục JSON: {JSON_OUTPUT_DIR}")
    
    for json_path in JSON_OUTPUT_DIR.glob("**/*.json"):
        if not json_path.is_file():
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            source_filename = data.get("source_filename", "unknown")
            data_type = data.get("data_type", "unknown")
            content = data.get("content")

            # --- Thông tin metadata từ đường dẫn file JSON ---
            relative_path = json_path.relative_to(JSON_OUTPUT_DIR)
            topic = str(relative_path.parent)
            if topic == ".": topic = "general"
            # Lấy tên file gốc (BaoCao.pdf -> BaoCao)
            document_id = relative_path.name.split('.')[0] 

            print(f"Processing JSON: {json_path.name} (topic: {topic}, id: {document_id})")

            # --- CHECK TRÙNG LẶP (Logic cũ của anh) ---
            if is_duplicate(topic, document_id):
                print(f"🧹 Xoá dữ liệu cũ: topic='{topic}', id='{document_id}'")
                vectorstore._collection.delete(where={"topic": topic, "document_id": document_id})

            if not content:
                print("  Bỏ qua (không có content).")
                continue

            # --- CHUNKING ROUTER (Dựa trên data_type) ---
            splits = []
            if data_type == "unstructured_doc":
                splits = chunk_unstructured_elements(content)
            elif data_type == "table_rows":
                splits = chunk_table_rows(content)
            elif data_type == "plain_text":
                splits = chunk_plain_text(content.get("content", ""))
            elif data_type == "code":
                splits = chunk_code(content)
            
            if not splits:
                print("  Không tạo được chunk nào.")
                continue

            # Gán metadata chung cho tất cả chunks
            for chunk in splits:
                chunk_metadata = chunk.metadata if chunk.metadata is not None else {}
                chunk_metadata.update({
                    "topic": topic, 
                    "document_id": document_id, 
                    "source": source_filename
                })
                chunk.metadata = chunk_metadata

            # Thêm vào vector store
            vectorstore.add_documents(splits)
            print(f"  -> Cập nhật xong: {len(splits)} chunks.")

        except Exception as e:
            print(f"  LỖI khi xử lý JSON {json_path.name}: {e}")

    vectorstore.persist()
    print(f"\nĐã lưu vectorstore tổng hợp tại: {store_path}")
    print("\n🎉 HOÀN TẤT GIAI ĐOẠN 2: INGEST VECTOR STORE")

if __name__ == "__main__":
    main()