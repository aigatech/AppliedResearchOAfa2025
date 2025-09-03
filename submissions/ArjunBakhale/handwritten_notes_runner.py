#!/usr/bin/env python3
"""
Handwritten Notes RAG Pipeline Runner

Main entry point for the handwritten notes RAG system that combines
PDF scanning, OCR processing, and vector search capabilities.

Usage:
    python handwritten_notes_runner.py [options]

Commands:
    python handwritten_notes_runner.py                    # Start interactive CLI
    python handwritten_notes_runner.py --scan             # Discover PDFs only
    python handwritten_notes_runner.py --index            # Index all PDFs
    python handwritten_notes_runner.py --search "query"   # Direct search
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(debug: bool = False):
    """Configure logging for the application."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("handwritten_notes.log")
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


def check_dependencies():
    """Check that required services are available."""
    import requests
    
    try:
        # Check Weaviate
        response = requests.get("http://localhost:8777/v1/.well-known/ready", timeout=5)
        if response.status_code != 200:
            print("FAIL Weaviate is not ready. Please start with: docker compose up weaviate -d")
            return False
    except requests.exceptions.RequestException:
        print("FAIL Weaviate is not running. Please start with: docker compose up weaviate -d")
        return False
    
    print("PASS Weaviate is running and ready")
    return True


def run_scan_mode(notes_dir: Path):
    """Run in scan-only mode."""
    from scraping.pdf_scanner import PDFScanner
    
    scanner = PDFScanner(notes_dir)
    pdfs = scanner.discover_pdfs()
    stats = scanner.get_processing_stats()
    
    print(f"Discovered {stats['total_pdfs']} PDFs in {notes_dir}")
    print(f"Total pages: {stats['total_pages']}")
    print(f"Total size: {stats['total_size_mb']} MB")
    
    for pdf_path in pdfs:
        print(f"  • {pdf_path.relative_to(notes_dir)}")


def run_index_mode(notes_dir: Path, weaviate_url: str):
    """Run in index-only mode."""
    from scraping.pdf_scanner import PDFScanner
    from scraping.handwriting_ocr import HandwritingOCRProcessor
    from indexing.notes_indexer import HandwrittenNotesIndexer
    
    print("Scanning for PDFs...")
    scanner = PDFScanner(notes_dir)
    pdfs = scanner.discover_pdfs()
    
    if not pdfs:
        print("No PDFs found to process")
        return
    
    print(f"Processing {len(pdfs)} PDFs with OCR...")
    ocr_processor = HandwritingOCRProcessor()
    
    all_results = []
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"  Processing {pdf_path.name} ({i}/{len(pdfs)})...")
        try:
            page_images = list(scanner.extract_pages_as_images(pdf_path))
            ocr_results = ocr_processor.process_pdf_pages(pdf_path, page_images)
            all_results.extend(ocr_results)
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    print(f"Indexing {len(all_results)} pages in Weaviate...")
    indexer = HandwrittenNotesIndexer(weaviate_url)
    indexed_count = indexer.index_ocr_results(all_results)
    
    print(f"Successfully indexed {indexed_count} pages")


def run_search_mode(query: str, notes_dir: Path, weaviate_url: str):
    """Run in search-only mode."""
    from indexing.notes_indexer import HandwrittenNotesIndexer
    
    indexer = HandwrittenNotesIndexer(weaviate_url)
    results = indexer.search_notes(query, limit=5)
    
    if not results:
        print(f"No results found for: {query}")
        return
    
    print(f"Found {len(results)} results for: {query}\n")
    
    for i, result in enumerate(results, 1):
        text = result.get("text", "")
        preview = text[:150] + "..." if len(text) > 150 else text
        
        print(f"{i}. {result.get('document_title', 'Unknown')}")
        print(f"   Course: {result.get('course', 'Unknown')} | Page: {result.get('page_number', '?')}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")
        print(f"   Preview: {preview}")
        print()


def run_interactive_mode(notes_dir: Path, weaviate_url: str):
    """Run in interactive CLI mode."""
    from cli.notes_cli import NotesRAGCLI
    
    cli = NotesRAGCLI(notes_dir, weaviate_url)
    cli.start()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Handwritten Notes RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python handwritten_notes_runner.py                          # Interactive mode
  python handwritten_notes_runner.py --scan                   # Scan PDFs only
  python handwritten_notes_runner.py --index                  # Index all PDFs
  python handwritten_notes_runner.py --search "linear algebra" # Search directly
        """
    )
    
    # Directory and service options
    parser.add_argument("--notes-dir", type=Path, default="myNotes", 
                       help="Path to notes directory (default: myNotes)")
    parser.add_argument("--weaviate-url", default="http://localhost:8777",
                       help="Weaviate database URL (default: http://localhost:8777)")
    
    # Mode options
    parser.add_argument("--scan", action="store_true",
                       help="Scan and discover PDFs only")
    parser.add_argument("--index", action="store_true", 
                       help="Index all discovered PDFs")
    parser.add_argument("--search", type=str,
                       help="Search handwritten notes directly")
    
    # General options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Validate notes directory
    if not args.notes_dir.exists():
        print(f"Notes directory not found: {args.notes_dir}")
        print(f"   Please create the directory or specify correct path with --notes-dir")
        sys.exit(1)
    
    # Check dependencies (unless just scanning)
    if not args.scan:
        if not check_dependencies():
            sys.exit(1)
    
    try:
        # Route to appropriate mode
        if args.scan:
            run_scan_mode(args.notes_dir)
        elif args.index:
            run_index_mode(args.notes_dir, args.weaviate_url)
        elif args.search:
            run_search_mode(args.search, args.notes_dir, args.weaviate_url)
        else:
            run_interactive_mode(args.notes_dir, args.weaviate_url)
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()