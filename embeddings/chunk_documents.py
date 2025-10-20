# chunk_documents.py

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits documents into smaller chunks that can be easily vectorized.

    Args:
        documents (list[Document]): List of LangChain Documents to be split into chunks.

    Returns:
        list[Document]: A list of chunked Documents suitable for embedding.
    """
    print("splitting documents...")
    # Rules are quite technical so use bigger chunks with bigger overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(documents)
    
    # documents = _smart_chunk_documents(
    #     documents=documents,
    #     target_words=250, 
    #     overlap_words=50,
    # )
    
    print("Done splitting text!")
    

    import random
    print("\npost embeddings ------------------------\n")
    for doc in documents:
        if random.randint(1, 20) == 1:
            print(doc)
    
    return documents


# def _smart_chunk_documents(
#     documents: list[Document],
#     target_words=200,
#     overlap_words=50
# ) -> list[Document]:
#     """
#     Creates variable-length chunks based on paragraphs and headings.
#     Preserves paragraph integrity and includes optional metadata like headings.

#     Args:
#         documents (list[Document]): List of LangChain Documents.
#         target_words (int): Approximate number of words per chunk.
#         overlap_words (int): Number of words to overlap between chunks.

#     Returns:
#         list[Document]: List of chunked Documents.
#     """
#     chunks = []

#     heading_pattern = re.compile(r'^[A-Z][A-Za-z ]{0,50}$')  # simple heuristic for headings

#     for doc in documents:
#         text = doc.page_content
#         metadata = doc.metadata.copy()

#         # Split text into paragraphs (double line breaks)
#         paragraphs = text.split("\n\n")

#         current_chunk = ""
#         current_word_count = 0

#         for para in paragraphs:
#             para_word_count = len(para.split())
            
#             if current_word_count + para_word_count > target_words and current_chunk:
#                 # finalize current chunk
#                 chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata.copy()))
                
#                 # start new chunk with overlap
#                 overlap = " ".join(current_chunk.split()[-overlap_words:])
#                 current_chunk = overlap + " " + para
#                 current_word_count = len(current_chunk.split())
#             else:
#                 current_chunk += " " + para
#                 current_word_count += para_word_count
                
#             if heading_pattern.match(para.strip()):
#                 metadata["heading"] = para.strip()


#         # add last chunk
#         if current_chunk.strip():
#             chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata.copy()))

#     return chunks
