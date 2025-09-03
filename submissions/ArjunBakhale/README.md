# Handwritten Notes RAG System

Extract text from handwritten PDFs and search through your notes using AI.

## What This Does

1. **Scans PDF files** in your `myNotes/` directory
2. **Extracts handwritten text** using OCR (Optical Character Recognition)
3. **Indexes content** in a vector database for semantic search
4. **Provides search interface** to find information across all your notes

## Hugging Face Models Used

**1. OCR Processing (`scraping/handwriting_ocr.py`)**
- **Model**: `allenai/olmOCR-7B-0225-preview` (~20GB)
- **Components**:
  - `AutoProcessor.from_pretrained()` - Image preprocessing
  - `AutoTokenizer.from_pretrained()` - Text tokenization  
  - `AutoModelForVision2Seq.from_pretrained()` - Vision-to-text model
- **Purpose**: Converts handwritten PDF pages to text

**2. Search/Embeddings (`indexing/notes_indexer.py`)**
- **Model**: `BAAI/bge-m3` (~2GB)
- **Component**: `SentenceTransformer(model_name)`
- **Purpose**: Converts text to vectors for semantic search

**Total Download**: ~22GB on first use (models are cached locally)

**Purpose**
1. Apple notes and other free notetaking app users don't have the same search features as paid apps like GoodNotes
2. Even paid services like goodnotes don't support semantic search on all notes 
3. This allows natural language search for ideas within notes even if key words don't exactly match
4. CLI interface allows quick spin-up from the terminal

## Quick Setup

### 1. Install Dependencies

```bash
# Install pixi (package manager)
curl -fsSL https://pixi.sh/install.sh | bash

# Install project dependencies
pixi install
```

### 2. Install Docker

Download and install [Docker Desktop](https://docs.docker.com/desktop/install/mac-install/)

### 3. Start Database

```bash
# Start Weaviate vector database
docker compose up weaviate -d

# Verify it's running
curl http://localhost:8777/v1/.well-known/ready
```

### 4. Prepare Your Notes

Create a `myNotes/` directory and add your PDF files:
```
myNotes/
├── Math101/
│   ├── lecture1.pdf
│   └── homework.pdf
└── Physics/
    └── notes.pdf
```

## Usage

### Simple Text Extraction (Fast)

```bash
pixi shell
python basic_text_extractor.py
```
This creates `extracted_text_simple.txt` with any built-in text from PDFs.

### Full OCR Processing (Slow but accurate)

```bash
pixi shell
python simple_text_extractor.py
```
This uses AI OCR to read handwritten content and saves to `extracted_handwritten_text.txt`.

### Interactive Search System

```bash
pixi shell
python handwritten_notes_runner.py
```

Commands:
- `scan` - Find all PDFs
- `index` - Process PDFs with OCR and index for search
- `search "your query"` - Search through indexed content
- `debug` - Show what content is indexed
- `help` - Show all commands

## Troubleshooting

**Weaviate not running:**
```bash
docker compose up weaviate -d
```

**OCR taking too long:**
- Use `basic_text_extractor.py` first to see if PDFs have built-in text
- For handwritten content, the OCR model is large and takes time to download

**No search results:**
- Run `debug` command to see what's indexed
- Make sure indexing completed successfully
- Try broader search terms


