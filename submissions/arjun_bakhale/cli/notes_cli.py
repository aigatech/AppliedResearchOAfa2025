"""
CLI interface for handwritten notes RAG system.

Provides interactive commands for indexing, searching, and managing
handwritten notes extracted from PDFs.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

logger = logging.getLogger(__name__)


class NotesRAGCLI:
    """
    Command-line interface for handwritten notes RAG system.
    """
    
    def __init__(self, notes_directory: Path = None, weaviate_url: str = "http://localhost:8777"):
        """
        Initialize CLI.
        
        Args:
            notes_directory: Path to myNotes directory
            weaviate_url: Weaviate database URL
        """
        self.notes_directory = notes_directory or Path("myNotes")
        self.weaviate_url = weaviate_url
        self.console = Console()
        self.running = False
        
        # Initialize components (lazy loading)
        self._scanner = None
        self._ocr_processor = None  
        self._indexer = None
    
    @property
    def scanner(self):
        """Lazy load PDF scanner."""
        if self._scanner is None:
            from scraping.pdf_scanner import PDFScanner
            self._scanner = PDFScanner(self.notes_directory)
        return self._scanner
    
    @property
    def ocr_processor(self):
        """Lazy load OCR processor."""
        if self._ocr_processor is None:
            from scraping.handwriting_ocr import HandwritingOCRProcessor
            self._ocr_processor = HandwritingOCRProcessor()
        return self._ocr_processor
    
    @property
    def indexer(self):
        """Lazy load indexer."""
        if self._indexer is None:
            from indexing.notes_indexer import HandwrittenNotesIndexer
            self._indexer = HandwrittenNotesIndexer(self.weaviate_url)
        return self._indexer
    
    def start(self):
        """Start the interactive CLI session."""
        self.running = True
        
        # Welcome message
        self.console.print(Panel.fit(
            "[bold blue]Handwritten Notes RAG System[/bold blue]\n"
            f"Notes Directory: {self.notes_directory}\n"
            f"Weaviate URL: {self.weaviate_url}\n\n"
            "Type 'help' for available commands or 'exit' to quit.",
            title="Welcome"
        ))
        
        # Main REPL loop
        while self.running:
            try:
                user_input = Prompt.ask("[bold green]notes-rag[/bold green]").strip()
                
                if not user_input:
                    continue
                
                self._process_command(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' to quit gracefully[/yellow]")
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
        
        self.console.print("[bold blue]Goodbye![/bold blue]")
    
    def _process_command(self, user_input: str):
        """Process user command."""
        parts = user_input.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Command routing
        if command in ["help", "h"]:
            self._show_help()
        elif command in ["exit", "quit", "q"]:
            self.running = False
        elif command in ["scan", "discover"]:
            self._scan_pdfs()
        elif command in ["index", "process"]:
            pdf_path = args[0] if args else None
            self._index_pdfs(pdf_path)
        elif command in ["search", "s"]:
            query = " ".join(args) if args else Prompt.ask("Enter search query")
            self._search_notes(query)
        elif command in ["stats", "status"]:
            self._show_stats()
        elif command in ["list", "ls"]:
            self._list_indexed_documents()
        elif command in ["delete", "rm"]:
            pdf_path = args[0] if args else Prompt.ask("Enter PDF path to delete")
            self._delete_pdf(pdf_path)
        elif command in ["schema"]:
            action = args[0] if args else "create"
            self._manage_schema(action)
        elif command in ["debug", "show"]:
            self._debug_indexed_content()
        else:
            # Treat as search query
            self._search_notes(user_input)
    
    def _show_help(self):
        """Show available commands."""
        help_table = Table(title="Available Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        help_table.add_column("Example", style="dim")
        
        commands = [
            ("scan", "Discover PDFs in myNotes directory", "scan"),
            ("index [pdf_path]", "Index PDFs or specific PDF", "index myNotes/Linear/notes.pdf"),
            ("search <query>", "Search handwritten notes", "search linear algebra"),
            ("stats", "Show indexing statistics", "stats"),
            ("list", "List indexed documents", "list"),
            ("delete <pdf_path>", "Delete PDF from index", "delete myNotes/notes.pdf"),
            ("schema [create|delete]", "Manage Weaviate schema", "schema create"),
            ("debug", "Show indexed content for debugging", "debug"),
            ("help", "Show this help", "help"),
            ("exit", "Exit the application", "exit")
        ]
        
        for cmd, desc, example in commands:
            help_table.add_row(cmd, desc, example)
        
        self.console.print(help_table)
        self.console.print("\n[dim]You can also just type search queries directly without 'search' command[/dim]")
    
    def _scan_pdfs(self):
        """Scan and display discovered PDFs."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Scanning PDFs...", total=None)
                
                pdfs = self.scanner.discover_pdfs()
                stats = self.scanner.get_processing_stats()
                
                progress.remove_task(task)
            
            # Display results
            table = Table(title=f"Discovered PDFs in {self.notes_directory}")
            table.add_column("PDF", style="cyan")
            table.add_column("Pages", justify="right")
            table.add_column("Size", justify="right")
            table.add_column("Path", style="dim")
            
            for pdf_path in pdfs:
                try:
                    metadata = self.scanner.get_pdf_metadata(pdf_path)
                    table.add_row(
                        metadata["file_name"],
                        str(metadata.get("page_count", "?")),
                        f"{metadata.get('file_size', 0) / 1024:.1f} KB",
                        metadata.get("relative_path", "")
                    )
                except Exception:
                    table.add_row(pdf_path.name, "?", "?", str(pdf_path))
            
            self.console.print(table)
            self.console.print(f"\n[bold]Summary:[/bold] {stats['total_pdfs']} PDFs, {stats['total_pages']} pages, {stats['total_size_mb']} MB")
            
        except Exception as e:
            self.console.print(f"[red]Error scanning PDFs: {e}[/red]")
    
    def _index_pdfs(self, pdf_path: Optional[str] = None):
        """Index PDFs using OCR."""
        try:
            if pdf_path:
                # Index specific PDF
                pdf_file = Path(pdf_path)
                if not pdf_file.exists():
                    self.console.print(f"[red]PDF not found: {pdf_path}[/red]")
                    return
                
                pdfs_to_process = [pdf_file]
            else:
                # Index all PDFs
                pdfs_to_process = self.scanner.discover_pdfs()
            
            if not pdfs_to_process:
                self.console.print("[yellow]No PDFs found to process[/yellow]")
                return
            
            # Confirm before processing
            if len(pdfs_to_process) > 1:
                if not Confirm.ask(f"Process {len(pdfs_to_process)} PDFs?"):
                    return
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                all_results = []
                
                for i, pdf_path in enumerate(pdfs_to_process):
                    task = progress.add_task(f"Processing {pdf_path.name} ({i+1}/{len(pdfs_to_process)})...", total=None)
                    
                    try:
                        # Extract pages
                        page_images = list(self.scanner.extract_pages_as_images(pdf_path))
                        
                        # Process with OCR
                        ocr_results = self.ocr_processor.process_pdf_pages(pdf_path, page_images)
                        all_results.extend(ocr_results)
                        
                        progress.remove_task(task)
                        
                    except Exception as e:
                        progress.remove_task(task)
                        self.console.print(f"[red]Error processing {pdf_path.name}: {e}[/red]")
                        continue
                
                # Index results
                if all_results:
                    task = progress.add_task("Indexing in Weaviate...", total=None)
                    indexed_count = self.indexer.index_ocr_results(all_results)
                    progress.remove_task(task)
                    
                    self.console.print(f"[green]Successfully indexed {indexed_count} pages[/green]")
                else:
                    self.console.print("[yellow]No content extracted to index[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]Error during indexing: {e}[/red]")
    
    def _search_notes(self, query: str):
        """Search handwritten notes."""
        try:
            if not query.strip():
                self.console.print("[yellow]Empty search query[/yellow]")
                return
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Searching for: {query}", total=None)
                
                results = self.indexer.search_notes(query, limit=10)
                progress.remove_task(task)
            
            if not results:
                self.console.print("[yellow]No results found[/yellow]")
                return
            
            # Display results
            self.console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")
            
            for i, result in enumerate(results, 1):
                # Extract text preview
                text = result.get("text", "")
                preview = text[:200] + "..." if len(text) > 200 else text
                
                self.console.print(Panel(
                    f"[bold]{result.get('document_title', 'Unknown')}[/bold]\n"
                    f"Course: {result.get('course', 'Unknown')} | Unit: {result.get('unit', 'Unknown')}\n"
                    f"Page: {result.get('page_number', '?')} | Confidence: {result.get('confidence', 0):.2f}\n\n"
                    f"[dim]{preview}[/dim]",
                    title=f"Result {i}",
                    border_style="blue"
                ))
            
        except Exception as e:
            self.console.print(f"[red]Error searching: {e}[/red]")
    
    def _show_stats(self):
        """Show indexing statistics."""
        try:
            stats = self.indexer.get_stats()
            
            if "error" in stats:
                self.console.print(f"[red]Error getting stats: {stats['error']}[/red]")
                return
            
            stats_table = Table(title="Indexing Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            stats_table.add_row("Total Documents", str(stats.get("total_documents", 0)))
            stats_table.add_row("Unique PDFs", str(stats.get("unique_pdfs", 0)))
            stats_table.add_row("Unique Courses", str(stats.get("unique_courses", 0)))
            stats_table.add_row("Average Confidence", f"{stats.get('average_confidence', 0):.3f}")
            stats_table.add_row("Last Indexed", stats.get("last_indexed", "Never"))
            
            self.console.print(stats_table)
            
        except Exception as e:
            self.console.print(f"[red]Error getting statistics: {e}[/red]")
    
    def _list_indexed_documents(self):
        """List all indexed documents."""
        try:
            documents = self.indexer.get_indexed_documents()
            
            if not documents:
                self.console.print("[yellow]No indexed documents found[/yellow]")
                return
            
            table = Table(title="Indexed Documents")
            table.add_column("Document", style="cyan")
            table.add_column("Course", style="green")
            table.add_column("Unit", style="blue")
            table.add_column("Page", justify="right")
            table.add_column("Confidence", justify="right")
            table.add_column("Indexed", style="dim")
            
            for doc in documents:
                table.add_row(
                    doc.get("document_title", "Unknown"),
                    doc.get("course", "Unknown"),
                    doc.get("unit", "Unknown"),
                    str(doc.get("page_number", "?")),
                    f"{doc.get('confidence', 0):.2f}",
                    doc.get("indexed_at", "Unknown")[:10]  # Just date part
                )
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Error listing documents: {e}[/red]")
    
    def _delete_pdf(self, pdf_path: str):
        """Delete a PDF from the index."""
        try:
            if Confirm.ask(f"Delete all pages from {pdf_path}?"):
                deleted_count = self.indexer.delete_by_pdf_path(pdf_path)
                self.console.print(f"[green]Deleted {deleted_count} documents[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error deleting: {e}[/red]")
    
    def _manage_schema(self, action: str):
        """Manage Weaviate schema."""
        try:
            if action == "create":
                if Confirm.ask("Create/recreate Weaviate schema?"):
                    self.indexer.create_schema(delete_existing=True)
                    self.console.print("[green]Schema created successfully[/green]")
            
            elif action == "delete":
                if Confirm.ask("Delete Weaviate schema? This will remove all indexed data."):
                    # Delete schema by recreating without data
                    self.indexer.create_schema(delete_existing=True)
                    self.console.print("[green]Schema deleted[/green]")
            
            else:
                self.console.print(f"[red]Unknown schema action: {action}[/red]")
                
        except Exception as e:
            self.console.print(f"[red]Error managing schema: {e}[/red]")
    
    def _debug_indexed_content(self):
        """Debug method to show what's actually indexed."""
        try:
            self.indexer.debug_show_all_indexed_content(limit=10)
        except Exception as e:
            self.console.print(f"[red]Debug error: {e}[/red]")


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Handwritten Notes RAG CLI")
    parser.add_argument("--notes-dir", type=Path, default="myNotes", help="Path to notes directory")
    parser.add_argument("--weaviate-url", default="http://localhost:8777", help="Weaviate database URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s - %(message)s",
        handlers=[logging.FileHandler("notes_cli.log")]
    )
    
    # Start CLI
    cli = NotesRAGCLI(args.notes_dir, args.weaviate_url)
    cli.start()


if __name__ == "__main__":
    main()