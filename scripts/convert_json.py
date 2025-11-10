# run_parser.py
# (Giai đoạn 1: Parse file thô -> file JSON)

import os
import sys
import json
from pathlib import Path

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import SOURCE_DOCS_DIR, JSON_OUTPUT_DIR
# Import thư viện parser mới của chúng ta
from src.chatbot.core import document_processing as parser

# --- Định nghĩa các loại file ---
CODE_EXTENSIONS = {".py", ".js", ".java", ".md", ".html", ".css"}
DOC_EXTENSIONS = {".pdf", ".docx"} # .txt sẽ xử lý riêng
TXT_EXTENSIONS = {".txt"}
CSV_EXTENSIONS = {".csv"}
XLSX_EXTENSIONS = {".xlsx"}
STRUCTURED_TEXT_EXTENSIONS = {".dat"} # file data xe của anh

def main():
    print("--- BẮT ĐẦU GIAI ĐOẠN 1: PARSE FILES SANG JSON ---")
    JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    processed_files = 0
    for file_path in SOURCE_DOCS_DIR.glob("**/*"):
        if not file_path.is_file():
            continue
        
        ext = file_path.suffix.lower()
        
        # --- Xác định đường dẫn output ---
        # data/source_docs/folder/file.pdf -> data/json_output/folder/file.pdf.json
        relative_path = file_path.relative_to(SOURCE_DOCS_DIR)
        json_output_path = JSON_OUTPUT_DIR / relative_path.with_suffix(".json")
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # --- Logic: Bỏ qua nếu file JSON đã tồn tại và mới hơn file gốc ---
        if json_output_path.exists():
            json_mod_time = json_output_path.stat().st_mtime
            file_mod_time = file_path.stat().st_mtime
            if json_mod_time > file_mod_time:
                # print(f"  Skipping (JSON is up-to-date): {file_path.name}")
                continue

        print(f"Processing: {file_path.name}")
        parsed_data = None
        data_type = "unknown" # Dùng để báo cho Giai đoạn 2 biết cách chunk

        try:
            # --- LOGIC ĐỊNH TUYẾN (ROUTING) ---
            if ext in DOC_EXTENSIONS:
                data_type = "unstructured_doc" # PDF, DOCX
                if ext == ".pdf":
                    parsed_data = parser.parse_pdf(file_path)
                elif ext == ".docx":
                    parsed_data = parser.parse_docx(file_path)
            
            elif ext in TXT_EXTENSIONS:
                data_type = "plain_text"
                parsed_data = parser.parse_text(file_path)
            
            elif ext in CODE_EXTENSIONS:
                data_type = "code"
                parsed_data = parser.parse_code(file_path)
            
            elif ext in CSV_EXTENSIONS:
                data_type = "table_rows" # Dữ liệu dạng hàng
                parsed_data = parser.parse_csv(file_path)
            
            elif ext in STRUCTURED_TEXT_EXTENSIONS:
                data_type = "table_rows" # Dữ liệu dạng hàng
                parsed_data = parser.parse_structured_text(file_path)
            elif ext in XLSX_EXTENSIONS:
                data_type = "table_rows" # Dữ liệu dạng hàng
                parsed_data = parser.parse_excel(file_path)
            
            else:
                print(f"  Bỏ qua (không hỗ trợ ext): {file_path.name}")
                continue

            # --- Lưu file JSON ---
            if parsed_data:
                # Gói dữ liệu vào một object chuẩn
                output_json = {
                    "source_path": str(file_path),
                    "source_filename": file_path.name,
                    "data_type": data_type, # Rất quan trọng cho Giai đoạn 2
                    "content": parsed_data
                }
                
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_json, f, ensure_ascii=False, indent=2)
                processed_files += 1
                print(f"  -> Saved JSON: {json_output_path.name}")

        except Exception as e:
            print(f"  LỖI khi xử lý {file_path.name}: {e}")
    
    print(f"\n--- HOÀN TẤT GIAI ĐOẠN 1: Đã xử lý {processed_files} file mới. ---")

if __name__ == "__main__":
    main()