"""
PDF Scanner for recursive handwritten notes discovery.

Recursively scans the myNotes directory for PDF files and provides
utilities for extracting pages and metadata for OCR processing.
"""

import logging
import io
from pathlib import Path
from typing import List, Dict, Any, Generator, Tuple
import hashlib
from datetime import datetime

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


class PDFScanner:
    """
    Scanner for discovering and processing PDF files in myNotes directory.
    """
    
    def __init__(self, notes_directory: Path, target_dpi: int = 150):
        """
        Initialize PDF scanner.
        
        Args:
            notes_directory: Path to myNotes directory
            target_dpi: DPI for page rendering (150 works well for olmOCR)
        """
        self.notes_directory = Path(notes_directory)
        self.target_dpi = target_dpi
        
        if not self.notes_directory.exists():
            raise ValueError(f"Notes directory does not exist: {notes_directory}")
    
    def discover_pdfs(self) -> List[Path]:
        """
        Recursively discover all PDF files in the notes directory.
        
        Returns:
            List of PDF file paths
        """
        pdf_files = []
        
        try:
            for pdf_path in self.notes_directory.rglob("*.pdf"):
                if pdf_path.is_file():
                    pdf_files.append(pdf_path)
                    logger.debug(f"Found PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Error discovering PDFs: {e}")
            
        logger.debug(f"Discovered {len(pdf_files)} PDF files in {self.notes_directory}")
        return sorted(pdf_files)
    
    def get_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(pdf_path)
            
            # Get file stats
            file_stats = pdf_path.stat()
            
            return {
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "file_size": file_stats.st_size,
                "file_hash": file_hash,
                "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "page_count": doc.page_count,
                "title": metadata.get("title", pdf_path.stem),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "relative_path": str(pdf_path.relative_to(self.notes_directory))
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "error": str(e)
            }
        finally:
            if 'doc' in locals():
                doc.close()
    
    def extract_pages_as_images(self, pdf_path: Path) -> Generator[Tuple[int, Image.Image, Dict], None, None]:
        """
        Extract PDF pages as images for OCR processing.
        
        Args:
            pdf_path: Path to PDF file
            
        Yields:
            Tuple of (page_number, PIL_Image, page_metadata)
        """
        try:
            doc = fitz.open(pdf_path)
            logger.debug(f"Extracting {doc.page_count} pages from {pdf_path.name}")
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    
                    # Calculate matrix for target DPI
                    # PyMuPDF default is 72 DPI, so scale accordingly
                    zoom = self.target_dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    
                    # Render page as image
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Ensure longest dimension is 1024px for olmOCR
                    image = self._resize_for_olm_ocr(image)
                    
                    # Page metadata
                    page_metadata = {
                        "page_number": page_num + 1,
                        "width": pix.width,
                        "height": pix.height,
                        "dpi": self.target_dpi
                    }
                    
                    yield page_num + 1, image, page_metadata
                    
                except Exception as e:
                    logger.error(f"Error extracting page {page_num + 1} from {pdf_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path}: {e}")
            return
        finally:
            if 'doc' in locals():
                doc.close()
    
    def _resize_for_olm_ocr(self, image: Image.Image) -> Image.Image:
        """
        Resize image so longest dimension is 1024px as required by olmOCR.
        
        Args:
            image: PIL Image to resize
            
        Returns:
            Resized PIL Image
        """
        width, height = image.size
        max_dimension = max(width, height)
        
        if max_dimension <= 1024:
            return image
        
        # Calculate scale factor
        scale = 1024 / max_dimension
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file for deduplication.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about discovered PDFs.
        
        Returns:
            Dictionary with processing statistics
        """
        pdfs = self.discover_pdfs()
        total_pages = 0
        total_size = 0
        
        for pdf_path in pdfs:
            try:
                metadata = self.get_pdf_metadata(pdf_path)
                total_pages += metadata.get("page_count", 0)
                total_size += metadata.get("file_size", 0)
            except Exception:
                continue
        
        return {
            "total_pdfs": len(pdfs),
            "total_pages": total_pages,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "average_pages_per_pdf": round(total_pages / len(pdfs), 1) if pdfs else 0
        }


if __name__ == "__main__":
    import io
    
    # Demo usage
    notes_dir = Path("../myNotes")
    scanner = PDFScanner(notes_dir)
    
    # Print discovery stats
    stats = scanner.get_processing_stats()
    print(f"PDF Discovery Stats: {stats}")
    
    # Test page extraction on first PDF
    pdfs = scanner.discover_pdfs()
    if pdfs:
        first_pdf = pdfs[0]
        print(f"\nTesting page extraction on: {first_pdf.name}")
        
        for page_num, image, metadata in scanner.extract_pages_as_images(first_pdf):
            print(f"  Page {page_num}: {image.size} pixels")
            if page_num >= 2:  # Only test first 2 pages
                break