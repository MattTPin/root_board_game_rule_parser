# populate_db.py

import argparse
import os
import shutil

from langchain_core.documents import Document
from langchain_chroma import Chroma

from embeddings.get_embeddings import get_embedding_function
from embeddings.load_pdf import load_pdfs
from embeddings.chunk_documents import chunk_documents
from embeddings.clean_pdfs import clean_pdfs

CHROMA_PATH = "chroma"
BASE_PDF_PATH = "pdfs/"
CLEANED_PDF_PATH = "pdfs_cleaned/"
PDFS_TO_CLEAN = [
    'Root_Base_Learn_to_Play_web_Oct_15_2020.pdf'
]

def main():
    # Check if the database should be cleared (if users passes the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database before re-filling it.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Step 1: Clean PDFs first
    print("Checking and cleaning PDFs if needed...")
    clean_pdfs(
        input_folder = BASE_PDF_PATH,
        cleaned_folder = CLEANED_PDF_PATH,
        pdfs_to_clean = PDFS_TO_CLEAN
    )

    # ---- Load PDFs in parsable format
    documents = load_pdfs(CLEANED_PDF_PATH)
    
    chunks = chunk_documents(documents)
    
    # ---- Create (or update) the vectorDB.
    _add_to_chroma(chunks)


def _add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = _calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def _calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    # This will create IDs like "data/Root_Base_Law_of_Root.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
