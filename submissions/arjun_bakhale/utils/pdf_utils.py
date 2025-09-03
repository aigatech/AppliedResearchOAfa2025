"""
Utility functions for PDF processing and validation.

Common utilities shared across the handwritten notes RAG pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)


def validate_pdf_file(pdf_path: Path) -> bool:
    """
    Validate that a file is a valid PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if file is a valid PDF
    """
    try:
        if not pdf_path.exists():
            return False
        
        if pdf_path.suffix.lower() != '.pdf':
            return False
        
        # Check PDF signature
        with open(pdf_path, 'rb') as f:
            header = f.read(5)
            
        return header == b'%PDF-'
        
    except Exception as e:
        logger.debug(f"PDF validation failed for {pdf_path}: {e}")
        return False


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of a file.
    
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


def extract_course_info(pdf_path: Path, base_dir: Path) -> Tuple[str, str]:
    """
    Extract course and unit information from PDF path structure.
    
    Args:
        pdf_path: Path to PDF file
        base_dir: Base notes directory
        
    Returns:
        Tuple of (course, unit)
    """
    try:
        relative_path = pdf_path.relative_to(base_dir)
        path_parts = list(relative_path.parts[:-1])  # Exclude filename
        
        course = path_parts[0] if path_parts else "Unknown"
        unit = path_parts[1] if len(path_parts) > 1 else "General"
        
        return course, unit
        
    except Exception as e:
        logger.debug(f"Could not extract course info from {pdf_path}: {e}")
        return "Unknown", "General"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe processing.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace problematic characters
    safe_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    sanitized = "".join(c if c in safe_chars else "_" for c in filename)
    
    # Remove multiple underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    
    return sanitized.strip("_")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better vector search.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break on sentence boundaries
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end < len(text) else len(text)
        
        if start >= len(text):
            break
    
    return chunks


if __name__ == "__main__":
    # Demo utilities
    test_pdf = Path("../myNotes/Linear/Unit1/Week I Notes.pdf")
    
    if test_pdf.exists():
        print(f"Valid PDF: {validate_pdf_file(test_pdf)}")
        print(f"File hash: {calculate_file_hash(test_pdf)[:16]}...")
        
        course, unit = extract_course_info(test_pdf, Path("../myNotes"))
        print(f"Course: {course}, Unit: {unit}")
    
    # Test text chunking
    sample_text = "This is a sample text. " * 50
    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
    print(f"Text chunked into {len(chunks)} pieces")