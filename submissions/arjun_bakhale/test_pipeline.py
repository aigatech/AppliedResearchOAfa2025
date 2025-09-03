#!/usr/bin/env python3
"""
Quick test script for the handwritten notes RAG pipeline.

Tests basic functionality without full processing to verify setup.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from scraping.pdf_scanner import PDFScanner
        print("  PASS PDF Scanner")
    except Exception as e:
        print(f"  FAIL PDF Scanner: {e}")
        return False
    
    try:
        from scraping.handwriting_ocr import HandwritingOCRProcessor  
        print("  PASS OCR Processor")
    except Exception as e:
        print(f"  FAIL OCR Processor: {e}")
        return False
    
    try:
        from indexing.notes_indexer import HandwrittenNotesIndexer
        print("  PASS Notes Indexer")
    except Exception as e:
        print(f"  FAIL Notes Indexer: {e}")
        return False
    
    try:
        from cli.notes_cli import NotesRAGCLI
        print("  PASS CLI Interface")
    except Exception as e:
        print(f"  FAIL CLI Interface: {e}")
        return False
    
    try:
        from config import NotesRAGConfig
        print("  PASS Configuration")
    except Exception as e:
        print(f"  FAIL Configuration: {e}")
        return False
    
    return True


def test_pdf_discovery():
    """Test PDF discovery functionality."""
    print("\nTesting PDF discovery...")
    
    try:
        from scraping.pdf_scanner import PDFScanner
        
        notes_dir = Path("myNotes")
        if not notes_dir.exists():
            print(f"  FAIL Notes directory not found: {notes_dir}")
            return False
        
        scanner = PDFScanner(notes_dir)
        pdfs = scanner.discover_pdfs()
        stats = scanner.get_processing_stats()
        
        print(f"  PASS Found {stats['total_pdfs']} PDFs")
        print(f"  Total pages: {stats['total_pages']}")
        print(f"  Total size: {stats['total_size_mb']} MB")
        
        # Test metadata extraction on first PDF
        if pdfs:
            first_pdf = pdfs[0]
            metadata = scanner.get_pdf_metadata(first_pdf)
            print(f"  Sample PDF: {metadata['file_name']} ({metadata['page_count']} pages)")
        
        return True
        
    except Exception as e:
        print(f"  FAIL PDF discovery failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\nTesting configuration...")
    
    try:
        from config import NotesRAGConfig
        
        # Load default config
        config = NotesRAGConfig.load()
        print(f"  PASS Loaded config")
        print(f"  Notes dir: {config.get_notes_path()}")
        print(f"  OCR model: {config.models.ocr_model}")
        print(f"  Weaviate URL: {config.database.weaviate_url}")
        
        return True
        
    except Exception as e:
        print(f"  FAIL Configuration failed: {e}")
        return False


def test_weaviate_connection():
    """Test Weaviate database connection."""
    print("\nTesting Weaviate connection...")
    
    try:
        import requests
        
        response = requests.get("http://localhost:8777/v1/.well-known/ready", timeout=5)
        if response.status_code == 200:
            print("  PASS Weaviate is running and ready")
            return True
        else:
            print(f"  FAIL Weaviate returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  FAIL Weaviate connection failed: {e}")
        print("     Please start Weaviate with: docker compose up weaviate -d")
        return False


def main():
    """Run all tests."""
    print("Handwritten Notes RAG Pipeline Test\n")
    
    tests = [
        ("Imports", test_imports),
        ("PDF Discovery", test_pdf_discovery),
        ("Configuration", test_configuration),
        ("Weaviate Connection", test_weaviate_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\nTest Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("  1. Start Weaviate: docker compose up weaviate -d")
        print("  2. Install dependencies: pixi install && pixi shell")
        print("  3. Run pipeline: pixi run python handwritten_notes_runner.py")
    else:
        print(f"\n{total - passed} test(s) failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()