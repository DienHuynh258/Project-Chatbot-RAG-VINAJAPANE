# src/chatbot/core/document_processing.py
# (File này giờ CHỈ làm nhiệm vụ PARSE, không chunk)

import pandas as pd
import re
from pathlib import Path
from langchain_community.document_loaders import TextLoader
import pandas as pd
# Cần cài đặt: 
# pip install unstructured "unstructured[pdf]" "unstructured[docx]" pandas
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.documents.elements import Element

def _elements_to_dicts(elements: list[Element]) -> list[dict]:
    """Helper: Chuyển list Element của unstructured sang list dict"""
    output = []
    for el in elements:
        # Chuyển element sang dict
        el_dict = el.to_dict() 
        # Loại bỏ các trường không thể serialize sang JSON
        el_dict['metadata'].pop('text_as_html', None)
        el_dict['metadata'].pop('header_footer_type', None)
        el_dict['metadata'].pop('page_number', None)
        output.append(el_dict)
    return output

def parse_pdf(file_path: Path) -> list[dict]:
    """Phân vùng PDF và trả về list[dict] các elements"""
    print(f"  Parsing PDF: {file_path.name}")
    elements = partition_pdf(
        filename=str(file_path),
        strategy="auto",
        infer_table_structure=True,
        extract_images_in_pdf=False
    )
    return _elements_to_dicts(elements)

def parse_docx(file_path: Path) -> list[dict]:
    """Phân vùng DOCX và trả về list[dict] các elements"""
    print(f"  Parsing DOCX: {file_path.name}")
    elements = partition_docx(filename=str(file_path))
    return _elements_to_dicts(elements)

def parse_text(file_path: Path) -> dict:
    """Đọc file .txt đơn giản và trả về content"""
    print(f"  Parsing TXT: {file_path.name}")
    loader = TextLoader(str(file_path), encoding="utf-8")
    text = loader.load()[0].page_content
    return {"content": text}

def parse_code(file_path: Path) -> dict:
    """Đọc file code và trả về content + language"""
    print(f"  Parsing Code: {file_path.name}")
    language_map = {
        ".py": "python", ".js": "javascript", ".java": "java", ".md": "markdown",
        ".html": "html", ".css": "css"
    }
    ext = file_path.suffix.lower()
    lang = language_map.get(ext, "text")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    return {"language": lang, "content": code}

def parse_csv(file_path: Path) -> list[dict]:
    """Đọc CSV và trả về list các hàng (dạng dict)"""
    print(f"  Parsing CSV: {file_path.name}")
    df = pd.read_csv(file_path, encoding='utf-8', keep_default_na=False)
    return df.to_dict('records')

# --- XỬ LÝ EXCEL ---
def parse_excel(file_path: Path) -> list[dict]:
    """
    Đọc file Excel (mặc định lấy sheet đầu tiên) và trả về list các hàng.
    """
    print(f"  Parsing Excel: {file_path.name}")
    try:
        # sheet_name=0 là đọc sheet đầu tiên
        df = pd.read_excel(file_path, sheet_name=0, keep_default_na=False)
        # Chuyển đổi tất cả dữ liệu sang string để đảm bảo JSON serialize được
        df = df.astype(str) 
        return df.to_dict('records')
    except Exception as e:
        print(f"  Lỗi khi đọc Excel {file_path.name}: {e}")
        return []

def parse_structured_text(file_path: Path) -> list[dict]:
    """Đọc file .dat (whitespace-delimited) và trả về list các hàng"""
    print(f"  Parsing DAT: {file_path.name}")
    df = pd.read_csv(
        file_path, 
        encoding='utf-8', 
        sep=r'\s\s+',
        engine='python',
        keep_default_na=False
    )
    return df.to_dict('records')