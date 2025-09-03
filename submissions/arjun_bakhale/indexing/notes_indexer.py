"""
Notes indexer for handwritten content using Weaviate.

Adapts the local-rag Weaviate indexer for handwritten notes with
specialized schema and processing for OCR-extracted content.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import weaviate
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Schema for handwritten notes with additional OCR metadata
HANDWRITTEN_NOTES_SCHEMA: List[Dict] = [
    {
        "name": "text",
        "dataType": ["text"],
        "description": "OCR-extracted text content from handwritten notes",
        "tokenization": 'word'
    },
    {
        "name": "pdf_path",
        "dataType": ["text"], 
        "description": "Path to source PDF file",
        "tokenization": 'field'
    },
    {
        "name": "page_number",
        "dataType": ["int"],
        "description": "Page number within the PDF"
    },
    {
        "name": "confidence",
        "dataType": ["number"],
        "description": "OCR confidence score (0.0 to 1.0)"
    },
    {
        "name": "course",
        "dataType": ["text"],
        "description": "Course or subject extracted from file path",
        "tokenization": 'field'
    },
    {
        "name": "unit",
        "dataType": ["text"], 
        "description": "Unit or chapter extracted from file path",
        "tokenization": 'field'
    },
    {
        "name": "document_title",
        "dataType": ["text"],
        "description": "Title or filename of the document",
        "tokenization": 'word'
    },
    {
        "name": "ocr_method",
        "dataType": ["text"],
        "description": "OCR processing method used",
        "tokenization": 'field'
    },
    {
        "name": "indexed_at",
        "dataType": ["date"],
        "description": "Timestamp when document was indexed"
    },
    {
        "name": "file_hash",
        "dataType": ["text"],
        "description": "SHA-256 hash of source PDF for deduplication",
        "tokenization": 'field'
    },
    {
        "name": "image_size",
        "dataType": ["text"],
        "description": "Dimensions of processed image",
        "tokenization": 'field'
    }
]


class HandwrittenNotesIndexer:
    """
    Weaviate indexer specialized for handwritten notes content.
    
    Based on local-rag's weaviate_indexer but optimized for OCR-extracted
    handwritten content with additional metadata and confidence tracking.
    """
    
    def __init__(self, weaviate_url: str = "http://localhost:8777", class_name: str = "HandwrittenNotes"):
        """
        Initialize the indexer.
        
        Args:
            weaviate_url: Weaviate database URL
            class_name: Weaviate class name for handwritten notes
        """
        self.weaviate_url = weaviate_url
        self.class_name = class_name
        self.client: Optional[weaviate.Client] = None
        self.embedding_model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer for embeddings."""
        try:
            # Use the same embedding model as local-rag for consistency
            model_name = "BAAI/bge-m3"
            logger.debug(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            logger.debug("Successfully loaded embedding model")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def _get_client(self) -> weaviate.Client:
        """Get or create Weaviate client."""
        if self.client is None:
            self.client = weaviate.Client(self.weaviate_url)
            
            if not self.client.is_live():
                raise ConnectionError(f"Weaviate is not live at {self.weaviate_url}")
                
            if not self.client.is_ready():
                raise ConnectionError(f"Weaviate is not ready at {self.weaviate_url}")
                
        return self.client
    
    def create_schema(self, delete_existing: bool = False):
        """
        Create Weaviate schema for handwritten notes.
        
        Args:
            delete_existing: Whether to delete existing schema first
        """
        client = self._get_client()
        
        try:
            # Check if class already exists
            schema = client.schema.get()
            existing_classes = {c["class"] for c in schema.get("classes", [])}
            
            if self.class_name in existing_classes:
                if delete_existing:
                    logger.warning(f"Deleting existing class: {self.class_name}")
                    client.schema.delete_class(self.class_name)
                else:
                    logger.debug(f"Schema for {self.class_name} already exists")
                    return
            
            # Create new class
            class_obj = {
                "class": self.class_name,
                "description": "Handwritten notes extracted via OCR from PDF documents",
                "properties": HANDWRITTEN_NOTES_SCHEMA,
                "vectorizer": "none"  # We'll provide our own vectors
            }
            
            client.schema.create_class(class_obj)
            logger.debug(f"Created Weaviate schema for {self.class_name}")
            
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            raise RuntimeError(f"Failed to create Weaviate schema: {e}")
    
    def index_ocr_results(self, ocr_results: List[Dict[str, Any]], batch_size: int = 10) -> int:
        """
        Index OCR results into Weaviate.
        
        Args:
            ocr_results: List of OCR processing results
            batch_size: Number of documents to process in each batch
            
        Returns:
            Number of successfully indexed documents
        """
        client = self._get_client()
        indexed_count = 0
        
        # Ensure schema exists
        self.create_schema(delete_existing=False)
        
        # Process in batches
        for i in range(0, len(ocr_results), batch_size):
            batch = ocr_results[i:i + batch_size]
            
            try:
                indexed_count += self._index_batch(client, batch)
                logger.debug(f"Indexed batch {i//batch_size + 1}: {indexed_count}/{len(ocr_results)} documents")
                
            except Exception as e:
                logger.error(f"Error indexing batch {i//batch_size + 1}: {e}")
                continue
        
        logger.debug(f"Indexing complete: {indexed_count}/{len(ocr_results)} documents indexed")
        return indexed_count
    
    def _index_batch(self, client: weaviate.Client, batch: List[Dict[str, Any]]) -> int:
        """
        Index a batch of OCR results.
        
        Args:
            client: Weaviate client
            batch: Batch of OCR results to index
            
        Returns:
            Number of successfully indexed documents in this batch
        """
        batch_indexed = 0
        
        for result in batch:
            try:
                # Skip empty or error results
                text_content = result.get("text", "").strip()
                if not text_content or result.get("error"):
                    logger.debug(f"Skipping empty/error result for {result.get('pdf_path', 'unknown')}")
                    continue
                
                # Generate embedding
                embedding = self._generate_embedding(text_content)
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for {result.get('pdf_path', 'unknown')}")
                    continue
                
                # Prepare document data
                doc_data = self._prepare_document_data(result)
                
                # Index in Weaviate
                client.data_object.create(
                    data_object=doc_data,
                    class_name=self.class_name,
                    vector=embedding
                )
                
                batch_indexed += 1
                logger.debug(f"Indexed: {result.get('pdf_path', 'unknown')} page {result.get('page_number', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Error indexing document: {e}")
                continue
        
        return batch_indexed
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            if not self.embedding_model:
                logger.error("Embedding model not initialized")
                return None
            
            # Generate embedding
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def _prepare_document_data(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare OCR result for Weaviate indexing.
        
        Args:
            ocr_result: OCR processing result
            
        Returns:
            Document data for Weaviate
        """
        context = ocr_result.get("context", {})
        
        return {
            "text": ocr_result.get("text", ""),
            "pdf_path": ocr_result.get("pdf_path", ""),
            "page_number": ocr_result.get("page_number", 0),
            "confidence": float(ocr_result.get("confidence", 0.0)),
            "course": context.get("course", "Unknown"),
            "unit": context.get("unit", "Unknown"), 
            "document_title": context.get("file_name", ""),
            "ocr_method": ocr_result.get("method", "olmocr"),
            "indexed_at": datetime.now().isoformat(),
            "file_hash": context.get("file_hash", ""),
            "image_size": str(ocr_result.get("image_size", ""))
        }
    
    def search_notes(self, query: str, limit: int = 10, course_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search handwritten notes using semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            course_filter: Optional course name filter
            
        Returns:
            List of search results
        """
        try:
            client = self._get_client()
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Build search query
            search_query = client.query.get(
                self.class_name,
                ["text", "pdf_path", "page_number", "confidence", "course", "unit", "document_title"]
            ).with_near_vector({
                "vector": query_embedding
            }).with_limit(limit)
            
            # Add course filter if specified
            if course_filter:
                search_query = search_query.with_where({
                    "path": ["course"],
                    "operator": "Equal",
                    "valueText": course_filter
                })
            
            # Execute search
            result = search_query.do()
            
            # Extract results
            documents = []
            if "data" in result and "Get" in result["data"]:
                for doc in result["data"]["Get"][self.class_name]:
                    documents.append(doc)
            
            logger.debug(f"Search returned {len(documents)} results for query: {query}")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching notes: {e}")
            return []
    
    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Get all indexed handwritten notes documents.
        
        Returns:
            List of all indexed documents with metadata
        """
        try:
            client = self._get_client()
            
            result = client.query.get(
                self.class_name,
                ["pdf_path", "page_number", "confidence", "course", "unit", "document_title", "indexed_at", "file_hash"]
            ).with_limit(10000).do()
            
            documents = []
            if "data" in result and "Get" in result["data"]:
                for doc in result["data"]["Get"][self.class_name]:
                    documents.append(doc)
            
            logger.debug(f"Found {len(documents)} indexed handwritten notes documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error querying indexed documents: {e}")
            return []
    
    def delete_by_pdf_path(self, pdf_path: str) -> int:
        """
        Delete all pages from a specific PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Number of documents deleted
        """
        try:
            client = self._get_client()
            
            # Find all documents for this PDF
            result = client.query.get(
                self.class_name,
                ["pdf_path"]
            ).with_where({
                "path": ["pdf_path"],
                "operator": "Equal", 
                "valueText": str(pdf_path)
            }).with_additional(["id"]).do()
            
            deleted_count = 0
            if "data" in result and "Get" in result["data"]:
                for doc in result["data"]["Get"][self.class_name]:
                    doc_id = doc["_additional"]["id"]
                    client.data_object.delete(
                        uuid=doc_id,
                        class_name=self.class_name
                    )
                    deleted_count += 1
            
            logger.debug(f"Deleted {deleted_count} documents for PDF: {pdf_path}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting documents for {pdf_path}: {e}")
            return 0
    
    def debug_show_all_indexed_content(self, limit: int = 5):
        """
        Debug method to show what content is actually indexed.
        
        Args:
            limit: Number of documents to show
        """
        try:
            client = self._get_client()
            
            result = (
                client.query
                .get(self.class_name, ["text", "document_title", "page_number", "course", "confidence"])
                .with_limit(limit)
                .do()
            )
            
            documents = result.get("data", {}).get("Get", {}).get(self.class_name, [])
            
            print(f"\n=== INDEXED CONTENT DEBUG (showing {len(documents)} docs) ===")
            for i, doc in enumerate(documents, 1):
                print(f"\nDocument {i}:")
                print(f"  Title: {doc.get('document_title', 'N/A')}")
                print(f"  Course: {doc.get('course', 'N/A')}")
                print(f"  Page: {doc.get('page_number', 'N/A')}")
                print(f"  Confidence: {doc.get('confidence', 'N/A')}")
                text = doc.get('text', '')
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"  Text: {preview}")
            print("=== END DEBUG ===\n")
            
            return documents
            
        except Exception as e:
            print(f"Debug error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get indexing statistics.
        
        Returns:
            Dictionary with indexing stats
        """
        try:
            client = self._get_client()
            
            # Get document count
            result = client.query.aggregate(self.class_name).with_fields("meta { count }").do()
            
            total_docs = 0
            try:
                total_docs = result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
            except (KeyError, IndexError):
                pass
            
            # Get indexed documents for additional stats
            documents = self.get_indexed_documents()
            
            # Calculate stats
            unique_pdfs = len(set(doc.get("pdf_path", "") for doc in documents))
            unique_courses = len(set(doc.get("course", "") for doc in documents if doc.get("course")))
            avg_confidence = sum(doc.get("confidence", 0) for doc in documents) / len(documents) if documents else 0
            
            return {
                "total_documents": total_docs,
                "unique_pdfs": unique_pdfs,
                "unique_courses": unique_courses,
                "average_confidence": round(avg_confidence, 3),
                "last_indexed": max((doc.get("indexed_at", "") for doc in documents), default="Never")
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Demo usage
    indexer = HandwrittenNotesIndexer()
    
    # Create schema
    indexer.create_schema(delete_existing=True)
    
    # Example OCR result for testing
    sample_ocr_result = {
        "text": "Sample handwritten text content",
        "pdf_path": "/path/to/notes.pdf",
        "page_number": 1,
        "confidence": 0.85,
        "method": "olmocr",
        "context": {
            "course": "Linear Algebra",
            "unit": "Unit1",
            "file_name": "Week 1 Notes.pdf",
            "file_hash": "abc123"
        },
        "image_size": "(800, 1024)"
    }
    
    # Test indexing
    result = indexer.index_ocr_results([sample_ocr_result])
    print(f"Indexed {result} documents")
    
    # Test search
    results = indexer.search_notes("handwritten text")
    print(f"Search returned {len(results)} results")
    
    # Print stats
    stats = indexer.get_stats()
    print(f"Index stats: {stats}")