#!/usr/bin/env python3
"""
Simple text extractor for handwritten PDFs.
Just extracts text and saves to a text file - no indexing or complex processing.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def extract_all_text():
    """Extract text from all PDFs and save to a single text file."""
    from scraping.pdf_scanner import PDFScanner
    from scraping.handwriting_ocr import HandwritingOCRProcessor
    
    notes_dir = Path("myNotes")
    output_file = Path("extracted_handwritten_text.txt")
    
    if not notes_dir.exists():
        print(f"Notes directory not found: {notes_dir}")
        return
    
    print("Scanning for PDFs...")
    scanner = PDFScanner(notes_dir)
    pdfs = scanner.discover_pdfs()
    
    if not pdfs:
        print("No PDFs found")
        return
    
    print(f"Found {len(pdfs)} PDFs")
    ocr_processor = HandwritingOCRProcessor(use_lightweight=False)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("HANDWRITTEN NOTES EXTRACTION\n")
        f.write("="*50 + "\n\n")
        
        for pdf_path in pdfs:
            print(f"Processing: {pdf_path.name}")
            f.write(f"FILE: {pdf_path.name}\n")
            f.write("-" * 30 + "\n")
            
            try:
                page_images = list(scanner.extract_pages_as_images(pdf_path))
                
                for page_num, image, page_metadata in page_images:
                    context = {
                        "page_number": page_num,
                        "course": pdf_path.parent.name,
                        "file_name": pdf_path.name
                    }
                    
                    result = ocr_processor.process_image(image, context)
                    text = result.get("text", "")
                    
                    f.write(f"\nPage {page_num}:\n")
                    f.write(text)
                    f.write("\n" + "."*20 + "\n")
                    
                    print(f"  Page {page_num}: {len(text)} characters")
                
                f.write("\n" + "="*50 + "\n\n")
                
            except Exception as e:
                print(f"  Error: {e}")
                f.write(f"ERROR processing {pdf_path.name}: {e}\n\n")
    
    print(f"\nExtracted text saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    extract_all_text()