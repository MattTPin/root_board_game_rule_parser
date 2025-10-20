# clean_pdfs.py

import pymupdf
import os
import shutil


def clean_pdfs(input_folder: str, cleaned_folder: str, pdfs_to_clean: list[str]):
    """
    Process all PDFs in `input_folder`. 
    - If a file is in `pdfs_to_clean`, clean it using `clean_pdf`.
    - Otherwise, copy it directly to `cleaned_folder`.
    Skips files that already exist in cleaned_folder.
    """
    os.makedirs(cleaned_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".pdf"):
            continue

        input_path = os.path.join(input_folder, filename)
        cleaned_path = os.path.join(cleaned_folder, filename)

        # Skip if already cleaned/copied
        if os.path.exists(cleaned_path):
            print(f"✅ Already exists in cleaned folder: {filename}")
            continue

        if filename in pdfs_to_clean:
            print(f"🧹 Cleaning PDF: {filename}")
            clean_pdf(input_path, cleaned_path)
        else:
            print(f"📁 Skipping cleaning for {filename} — copying as-is.")
            shutil.copy2(input_path, cleaned_path)


def clean_pdf(input_path: str, output_path: str):
    """
    Load a PDF, remove hidden/small text and replace symbols, then save cleaned PDF.
    """
    doc = pymupdf.open(input_path)
    new_doc = pymupdf.open()  # empty PDF to copy pages into

    for page in doc:
        text_blocks = page.get_text("dict")["blocks"]
        clean_text = ""

        for block in text_blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    # Skip invisible / tiny text
                    if span.get("size", 0) < 5 or span.get("color", 0) == 16777215:  # white / very small
                        continue
                    clean_text += span.get("text", "") + " "
                clean_text += "\n"

        # Create new page with cleaned text
        new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_text((50, 50), clean_text, fontsize=12)

    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    print(f"✅ Saved cleaned PDF: {output_path}")
