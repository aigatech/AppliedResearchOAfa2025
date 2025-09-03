#!/usr/bin/env python3
"""
Basic text extractor using PyMuPDF's built-in text extraction.
Much faster than OCR for testing.
"""

import sys
from pathlib import Path
import fitz  # PyMuPDF

def extract_text_simple():
    """Extract text using PyMuPDF's built-in text extraction."""
    notes_dir = Path("myNotes")
    output_file = Path("extracted_text_simple.txt")
    
    if not notes_dir.exists():
        print(f"Notes directory not found: {notes_dir}")
        return
    
    pdf_files = list(notes_dir.rglob("*.pdf"))
    if not pdf_files:
        print("No PDFs found")
        return
    
    print(f"Found {len(pdf_files)} PDFs")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PDF TEXT EXTRACTION (Built-in)\n")
        f.write("="*50 + "\n\n")
        
        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            f.write(f"FILE: {pdf_path.name}\n")
            f.write(f"PATH: {pdf_path.relative_to(notes_dir)}\n")
            f.write("-" * 30 + "\n")
            
            try:
                doc = fitz.open(pdf_path)
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text = page.get_text()
                    
                    f.write(f"\nPage {page_num + 1}:\n")
                    if text.strip():
                        f.write(text)
                        print(f"  Page {page_num + 1}: {len(text.strip())} characters")
                    else:
                        f.write("[No text found - likely handwritten/image content]")
                        print(f"  Page {page_num + 1}: No text (handwritten)")
                    f.write("\n" + "."*20 + "\n")
                
                doc.close()
                f.write("\n" + "="*50 + "\n\n")
                
            except Exception as e:
                print(f"  Error: {e}")
                f.write(f"ERROR: {e}\n\n")
    
    print(f"\nText saved to: {output_file}")

if __name__ == "__main__":
    extract_text_simple()