"""
Configuration management for cross-ref-cli.
Loads settings from .env file with sensible defaults.
"""

import os
from pathlib import Path
from typing import Any


class Config:
    """Configuration class that loads settings from environment variables."""

    def __init__(self, env_file: str = ".env"):
        """
        Initialize configuration by loading from .env file.

        Args:
            env_file: Path to the .env file (default: .env)
        """
        self.load_env(env_file)

    def load_env(self, env_file: str) -> None:
        """
        Load environment variables from .env file.

        Args:
            env_file: Path to the .env file
        """
        env_path = Path(env_file)
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value

    def get(self, key: str, default: Any = None, cast_type: type = str) -> Any:
        """
        Get a configuration value from environment variables.

        Args:
            key: The environment variable name
            default: Default value if not found
            cast_type: Type to cast the value to

        Returns:
            The configuration value cast to the specified type
        """
        value = os.environ.get(key, default)

        if value is None:
            return None

        # Handle boolean casting
        if cast_type == bool:
            if isinstance(value, bool):
                return value
            return value.lower() in ('true', '1', 'yes', 'on')

        # Handle other types
        try:
            return cast_type(value)
        except (ValueError, TypeError):
            return default

    # Model Configuration
    @property
    def embedding_model(self) -> str:
        return self.get('EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')

    # Chunking Configuration
    @property
    def chunk_size(self) -> int:
        return self.get('CHUNK_SIZE', 512, int)

    @property
    def chunk_overlap(self) -> int:
        return self.get('CHUNK_OVERLAP', 50, int)

    # Processing Configuration
    @property
    def batch_size(self) -> int:
        return self.get('BATCH_SIZE', 32, int)

    @property
    def max_tokens(self) -> int:
        return self.get('MAX_TOKENS', 512, int)

    @property
    def device(self) -> str:
        return self.get('DEVICE', 'cpu')

    @property
    def normalize_embeddings(self) -> bool:
        return self.get('NORMALIZE_EMBEDDINGS', True, bool)

    # Comparison Configuration
    @property
    def similarity_threshold(self) -> float:
        return self.get('SIMILARITY_THRESHOLD', 0.7, float)

    @property
    def top_k(self) -> int:
        return self.get('TOP_K', 5, int)

    # FAISS Configuration
    @property
    def index_type(self) -> str:
        return self.get('INDEX_TYPE', 'Flat')

    @property
    def n_clusters(self) -> int:
        return self.get('N_CLUSTERS', 100, int)

    # Output Configuration
    @property
    def include_chunk_text(self) -> bool:
        return self.get('INCLUDE_CHUNK_TEXT', True, bool)

    @property
    def max_chunk_display(self) -> int:
        return self.get('MAX_CHUNK_DISPLAY', 200, int)


# Global config instance
config = Config()
