# load_pdf.py

import re

from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader


def load_pdfs(data_path: str) -> list[Document]:
    """
    Load all PDF files as parsable text.
    """
    print("loading PDF documents...")
    document_loader = PyPDFDirectoryLoader(data_path)    
    documents = document_loader.load()
    print(f"Done loading {len(documents)} PDF documents!")

    # Clean each document's text
    for i, doc in enumerate(documents):
        if hasattr(doc, "page_content") and isinstance(doc.page_content, str):
            doc.page_content = _clean_pdf_text(doc.page_content)
        else:
            print(f"Warning: Document {i} has no page_content or it's not a string")
    
    import random
    print("\nPRE embeddings ------------------------\n")
    for doc in documents:
        if random.randint(1, 20) == 1:
            print(doc)
    
    return documents

def _clean_pdf_text(text: str) -> str:
    # 1. Remove hyphenation across lines and spaces (including weird mid-paragraph splits)
    text = re.sub(r'(\w+)\s*-\s+(\w+)', r'\1\2', text)

    # 2. Remove PDF artifact codes like "/zero.001" or "+/zero.001"
    text = re.sub(r'(\+)?(/\w+\.\d+)+', '', text)

    # 3. Remove isolated sequences of multiple slash-prefixed tokens (extra safety)
    text = re.sub(r'(?:/\w+\.\d+\s*)+', '', text)

    # 4. Replace newlines inside paragraphs with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # 6. Replace symbols with words (only standalone)
    text = _replace_root_pdf_symbols(text, ROOT_ITEM_SYMBOL_MAP)

    return text.strip()


ROOT_ITEM_SYMBOL_MAP = {
    "T": "[tea item]",
    "X": "[coin stack item]",
    "B": "[bag item]",
    "S": "[sword item]",
    "F": "[torch item]",
    "H": "[hammer item]",
    "C": "[crossbow item]",
    "M": "[boot item]"
    # add more as needed
}

def _replace_root_pdf_symbols(text: str, symbol_map: dict) -> str:
    """
    Replace symbol characters in PDFs with corresponding words,
    only when they appear as standalone tokens.
    """
    for symbol, word in symbol_map.items():
        # Replace only when symbol is surrounded by word boundaries
        text = re.sub(rf'\b{re.escape(symbol)}\b', word, text)
    return text