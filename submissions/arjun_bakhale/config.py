"""
Configuration management for handwritten notes RAG system.

Handles settings, paths, and model configurations with persistence.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".handwritten_notes_rag_config.json"


@dataclass
class ModelConfig:
    """Configuration for OCR and embedding models."""
    ocr_model: str = "allenai/olmOCR-7B-0225-preview"
    embedding_model: str = "BAAI/bge-m3"
    use_local_models: bool = True
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class ProcessingConfig:
    """Configuration for PDF processing and OCR."""
    target_dpi: int = 150
    max_image_dimension: int = 1024
    batch_size: int = 10
    confidence_threshold: float = 0.1
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class DatabaseConfig:
    """Configuration for Weaviate vector database."""
    weaviate_url: str = "http://localhost:8777"
    class_name: str = "HandwrittenNotes"
    index_timeout: int = 300
    search_limit: int = 10


@dataclass
class DirectoriesConfig:
    """Configuration for directory paths."""
    notes_directory: str = "myNotes"
    output_directory: str = "output"
    logs_directory: str = "logs"


@dataclass
class NotesRAGConfig:
    """Main configuration class for handwritten notes RAG system."""
    models: ModelConfig
    processing: ProcessingConfig
    database: DatabaseConfig
    directories: DirectoriesConfig
    debug: bool = False
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "NotesRAGConfig":
        """
        Load configuration from file or create default.
        
        Args:
            config_path: Path to config file, defaults to user home
            
        Returns:
            Configuration instance
        """
        config_path = config_path or DEFAULT_CONFIG_PATH
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                # Create config from loaded data
                config = cls(
                    models=ModelConfig(**data.get("models", {})),
                    processing=ProcessingConfig(**data.get("processing", {})),
                    database=DatabaseConfig(**data.get("database", {})),
                    directories=DirectoriesConfig(**data.get("directories", {})),
                    debug=data.get("debug", False)
                )
                
                logger.info(f"Loaded configuration from {config_path}")
                return config
                
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        # Return default configuration
        return cls(
            models=ModelConfig(),
            processing=ProcessingConfig(),
            database=DatabaseConfig(),
            directories=DirectoriesConfig()
        )
    
    def save(self, config_path: Optional[Path] = None):
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save config, defaults to user home
        """
        config_path = config_path or DEFAULT_CONFIG_PATH
        
        try:
            # Ensure parent directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            data = asdict(self)
            
            # Save to file
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise
    
    def get_notes_path(self) -> Path:
        """Get absolute path to notes directory."""
        return Path(self.directories.notes_directory).resolve()
    
    def get_output_path(self) -> Path:
        """Get absolute path to output directory.""" 
        return Path(self.directories.output_directory).resolve()
    
    def get_logs_path(self) -> Path:
        """Get absolute path to logs directory."""
        return Path(self.directories.logs_directory).resolve()
    
    def update_setting(self, key: str, value: Any) -> bool:
        """
        Update a configuration setting.
        
        Args:
            key: Setting key in dot notation (e.g., "models.ocr_model")
            value: New value
            
        Returns:
            True if setting was updated successfully
        """
        try:
            parts = key.split(".")
            
            if len(parts) == 2:
                section, setting = parts
                
                if section == "models":
                    setattr(self.models, setting, value)
                elif section == "processing":
                    setattr(self.processing, setting, value)
                elif section == "database":
                    setattr(self.database, setting, value)
                elif section == "directories":
                    setattr(self.directories, setting, value)
                else:
                    return False
                    
                logger.info(f"Updated setting {key} = {value}")
                return True
            else:
                # Top-level setting
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.info(f"Updated setting {key} = {value}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error updating setting {key}: {e}")
        
        return False
    
    def get_setting(self, key: str) -> Any:
        """
        Get a configuration setting value.
        
        Args:
            key: Setting key in dot notation
            
        Returns:
            Setting value or None if not found
        """
        try:
            parts = key.split(".")
            
            if len(parts) == 2:
                section, setting = parts
                
                if section == "models":
                    return getattr(self.models, setting, None)
                elif section == "processing":
                    return getattr(self.processing, setting, None)
                elif section == "database":
                    return getattr(self.database, setting, None)
                elif section == "directories":
                    return getattr(self.directories, setting, None)
            else:
                # Top-level setting
                return getattr(self, key, None)
                
        except Exception as e:
            logger.error(f"Error getting setting {key}: {e}")
        
        return None


if __name__ == "__main__":
    # Demo configuration usage
    config = NotesRAGConfig.load()
    
    print("Current configuration:")
    print(f"  Notes directory: {config.get_notes_path()}")
    print(f"  OCR model: {config.models.ocr_model}")
    print(f"  Weaviate URL: {config.database.weaviate_url}")
    
    # Test setting update
    config.update_setting("processing.target_dpi", 200)
    print(f"  Updated DPI: {config.processing.target_dpi}")
    
    # Save configuration
    config.save()
    print("Configuration saved")