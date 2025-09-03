"""
Handwriting OCR processor using olmOCR for extracting text from handwritten notes.

This module handles the integration with the olmOCR model from Allen AI
for processing handwritten content in PDF pages.
"""

import logging
import io
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import base64

from PIL import Image
import numpy as np
import torch

logger = logging.getLogger(__name__)


class HandwritingOCRProcessor:
    """
    OCR processor specialized for handwritten notes using olmOCR.
    """
    
    def __init__(self, model_name: str = "allenai/olmOCR-7B-0225-preview", use_lightweight: bool = True):
        """
        Initialize the OCR processor.
        
        Args:
            model_name: Hugging Face model identifier for olmOCR
            use_lightweight: Use lightweight OCR approach for faster processing
        """
        self.model_name = model_name
        self.use_lightweight = use_lightweight
        self.processor = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the olmOCR model and processor.
        """
        try:
            # Try using olmocr package first
            try:
                import olmocr
                self.ocr_engine = olmocr.OCREngine()
                self.use_olmocr_package = True
                return
            except ImportError:
                self.use_olmocr_package = False
            
            # Use lightweight mock OCR for development
            if self.use_lightweight:
                logger.debug("Using lightweight mock OCR for development")
                self.use_olmocr_package = False
                self.processor = None
                self.model = None
                return
            
            # Fallback to transformers implementation (slow)
            from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.debug(f"Using device: {self.device}")
            
            # Load model components
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            logger.debug("Successfully initialized olmOCR model with transformers")
            
        except Exception as e:
            logger.error(f"Failed to initialize olmOCR model: {e}")
            raise RuntimeError(f"Could not initialize olmOCR: {e}")
    
    def process_image(self, image: Image.Image, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a single image using olmOCR for handwriting recognition.
        
        Args:
            image: PIL Image to process
            context: Additional context (page number, file info, etc.)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            if self.use_olmocr_package:
                return self._process_with_olmocr_package(image, context)
            elif self.use_lightweight:
                return self._process_with_mock_ocr(image, context)
            else:
                return self._process_with_transformers(image, context)
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
                "context": context or {}
            }
    
    def _process_with_olmocr_package(self, image: Image.Image, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image using the olmocr package.
        
        Args:
            image: PIL Image to process
            context: Additional context
            
        Returns:
            Processing results
        """
        try:
            # Convert PIL image to format expected by olmocr
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Process with olmocr
            result = self.ocr_engine.process_image(img_buffer)
            
            extracted_text = result.get("text", "").strip()
            confidence = result.get("confidence", 1.0)
            
            logger.debug(f"olmOCR extracted {len(extracted_text)} characters with confidence {confidence}")
            
            return {
                "text": extracted_text,
                "confidence": confidence,
                "method": "olmocr_package",
                "context": context or {},
                "image_size": image.size
            }
            
        except Exception as e:
            logger.error(f"olmOCR package processing failed: {e}")
            raise
    
    def _process_with_transformers(self, image: Image.Image, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image using transformers implementation.
        
        Args:
            image: PIL Image to process
            context: Additional context
            
        Returns:
            Processing results
        """
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare inputs
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.1
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            ).strip()
            
            logger.debug(f"Transformers olmOCR extracted {len(generated_text)} characters")
            
            return {
                "text": generated_text,
                "confidence": 0.9,  # Default confidence for transformers approach
                "method": "transformers",
                "context": context or {},
                "image_size": image.size
            }
            
        except Exception as e:
            logger.error(f"Transformers processing failed: {e}")
            raise
    
    def _process_with_mock_ocr(self, image: Image.Image, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock OCR processor for development and testing.
        
        Args:
            image: PIL Image to process
            context: Additional context
            
        Returns:
            Mock processing results
        """
        # Generate mock handwritten text based on context
        page_num = context.get("page_number", 1) if context else 1
        course = context.get("course", "Unknown") if context else "Unknown"
        
        mock_text = f"Handwritten notes from {course} - Page {page_num}\n"
        mock_text += "Mathematics equations and diagrams\n"
        mock_text += "Key concepts and formulas\n"
        mock_text += "Student annotations and notes"
        
        return {
            "text": mock_text,
            "confidence": 0.85,
            "method": "mock_ocr",
            "context": context or {},
            "image_size": image.size
        }
    
    def process_pdf_pages(self, pdf_path: Path, page_images: List[Tuple[int, Image.Image, Dict]]) -> List[Dict[str, Any]]:
        """
        Process multiple PDF pages in batch.
        
        Args:
            pdf_path: Path to source PDF
            page_images: List of (page_number, image, metadata) tuples
            
        Returns:
            List of processing results for each page
        """
        results = []
        pdf_metadata = self.get_pdf_context(pdf_path)
        
        logger.debug(f"Processing {len(page_images)} pages from {pdf_path.name}")
        
        for page_num, image, page_metadata in page_images:
            try:
                # Combine PDF and page context
                context = {
                    **pdf_metadata,
                    **page_metadata,
                    "page_number": page_num
                }
                
                # Process the page
                result = self.process_image(image, context)
                result["pdf_path"] = str(pdf_path)
                result["page_number"] = page_num
                
                # Log progress and show extracted text
                text_length = len(result.get("text", ""))
                confidence = result.get("confidence", 0.0)
                extracted_text = result.get("text", "")
                
                logger.info(f"Page {page_num}: {text_length} chars, confidence: {confidence:.2f}")
                if extracted_text:
                    print(f"\n--- Page {page_num} Content ---")
                    print(extracted_text)
                    print("--- End Page Content ---\n")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing page {page_num} of {pdf_path}: {e}")
                results.append({
                    "pdf_path": str(pdf_path),
                    "page_number": page_num,
                    "text": "",
                    "confidence": 0.0,
                    "error": str(e),
                    "context": {"page_number": page_num}
                })
        
        return results
    
    def get_pdf_context(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Get contextual information about a PDF for processing.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Context dictionary
        """
        try:
            # Extract subject/topic from file path structure
            relative_path = pdf_path.relative_to(Path("myNotes"))
            path_parts = list(relative_path.parts[:-1])  # Exclude filename
            
            return {
                "file_name": pdf_path.name,
                "subject_path": "/".join(path_parts),
                "course": path_parts[0] if path_parts else "Unknown",
                "unit": path_parts[1] if len(path_parts) > 1 else "Unknown",
                "document_type": "handwritten_notes"
            }
        except Exception as e:
            logger.warning(f"Could not extract context from {pdf_path}: {e}")
            return {
                "file_name": pdf_path.name,
                "document_type": "handwritten_notes"
            }
    
    def batch_process_directory(self, notes_directory: Path) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory recursively.
        
        Args:
            notes_directory: Path to notes directory
            
        Returns:
            List of all processing results
        """
        from .pdf_scanner import PDFScanner
        
        scanner = PDFScanner(notes_directory)
        pdfs = scanner.discover_pdfs()
        
        all_results = []
        
        for pdf_path in pdfs:
            try:
                logger.debug(f"Processing PDF: {pdf_path}")
                
                # Extract pages as images
                page_images = list(scanner.extract_pages_as_images(pdf_path))
                
                # Process with OCR
                pdf_results = self.process_pdf_pages(pdf_path, page_images)
                all_results.extend(pdf_results)
                
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path}: {e}")
                continue
        
        logger.debug(f"Completed processing {len(pdfs)} PDFs, extracted text from {len(all_results)} pages")
        return all_results


if __name__ == "__main__":
    # Demo usage
    processor = HandwritingOCRProcessor()
    notes_dir = Path("../myNotes")
    
    if notes_dir.exists():
        results = processor.batch_process_directory(notes_dir)
        
        # Print summary
        total_text = sum(len(r.get("text", "")) for r in results)
        avg_confidence = np.mean([r.get("confidence", 0) for r in results])
        
        print(f"Processed {len(results)} pages")
        print(f"Total text extracted: {total_text} characters")
        print(f"Average confidence: {avg_confidence:.2f}")
    else:
        print(f"Notes directory not found: {notes_dir}")