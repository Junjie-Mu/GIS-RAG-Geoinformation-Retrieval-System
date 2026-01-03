import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except Exception:
        return default


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except Exception:
        return default


def _load_dotenv(dotenv_path: Path) -> Dict[str, str]:
    if not dotenv_path.exists():
        return {}

    data: Dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            data[key] = value
    return data


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV = _load_dotenv(_PROJECT_ROOT / ".env")


def _getenv(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, _DOTENV.get(key, default))


@dataclass(frozen=True)
class Settings:
    PROJECT_NAME: str
    VERSION: str
    DEBUG: bool

    API_HOST: str
    API_PORT: int
    WEB_HOST: str
    WEB_PORT: int

    DEVICE: str
    LLM_THINKING_MODEL_PATH: Optional[str]
    LLM_THINKING_MODEL_NAME: str
    LLM_INSTRUCT_MODEL_PATH: Optional[str]
    LLM_INSTRUCT_MODEL_NAME: str

    EMBEDDING_MODEL_PATH: Optional[str]
    EMBEDDING_MODEL_NAME: str
    EMBEDDING_MAX_LENGTH: int

    RETRIEVAL_MODE: str
    HYBRID_DENSE_WEIGHT: float
    HYBRID_LEXICAL_WEIGHT: float
    HYBRID_CANDIDATE_MULTIPLIER: int

    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    TOP_K: int
    SIMILARITY_THRESHOLD: float

    MAX_FILE_SIZE: int
    ALLOWED_EXTENSIONS: List[str]

    PROJECT_ROOT: Path
    DATA_DIR: Path
    PDF_DATA_DIR: Path
    VECTOR_DATA_DIR: Path
    RASTER_DATA_DIR: Path
    CHROMA_PERSIST_DIR: Path
    UPLOAD_DIR: Path
    LOGS_DIR: Path

def _build_settings() -> Settings:
    project_root = _PROJECT_ROOT
    data_dir = project_root / "data"

    api_host = _getenv("API_HOST", "127.0.0.1") or "127.0.0.1"
    web_host = _getenv("WEB_HOST", "127.0.0.1") or "127.0.0.1"
    embedding_model_path = (_getenv("EMBEDDING_MODEL_PATH") or "").strip() or None

    return Settings(
        PROJECT_NAME=_getenv("PROJECT_NAME", "GIS-RAG") or "GIS-RAG",
        VERSION=_getenv("VERSION", "1.0.0") or "1.0.0",
        DEBUG=_parse_bool(_getenv("DEBUG"), default=False),
        API_HOST=api_host,
        API_PORT=_parse_int(_getenv("API_PORT"), default=8000),
        WEB_HOST=web_host,
        WEB_PORT=_parse_int(_getenv("WEB_PORT"), default=8501),
        DEVICE=(_getenv("DEVICE", "cpu") or "cpu").strip().lower(),
        LLM_THINKING_MODEL_PATH=(_getenv("LLM_THINKING_MODEL_PATH") or "").strip() or None,
        LLM_THINKING_MODEL_NAME=_getenv("LLM_THINKING_MODEL_NAME", "qwen3-thinking") or "qwen3-thinking",
        LLM_INSTRUCT_MODEL_PATH=(_getenv("LLM_INSTRUCT_MODEL_PATH") or "").strip() or None,
        LLM_INSTRUCT_MODEL_NAME=_getenv("LLM_INSTRUCT_MODEL_NAME", "qwen3-instruct") or "qwen3-instruct",
        EMBEDDING_MODEL_PATH=embedding_model_path,
        EMBEDDING_MODEL_NAME=_getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        or "sentence-transformers/all-MiniLM-L6-v2",
        EMBEDDING_MAX_LENGTH=_parse_int(_getenv("EMBEDDING_MAX_LENGTH"), default=256),
        RETRIEVAL_MODE=(_getenv("RETRIEVAL_MODE", "dense") or "dense").strip().lower(),
        HYBRID_DENSE_WEIGHT=_parse_float(_getenv("HYBRID_DENSE_WEIGHT"), default=0.7),
        HYBRID_LEXICAL_WEIGHT=_parse_float(_getenv("HYBRID_LEXICAL_WEIGHT"), default=0.3),
        HYBRID_CANDIDATE_MULTIPLIER=_parse_int(_getenv("HYBRID_CANDIDATE_MULTIPLIER"), default=5),
        CHUNK_SIZE=_parse_int(_getenv("CHUNK_SIZE"), default=4000),
        CHUNK_OVERLAP=_parse_int(_getenv("CHUNK_OVERLAP"), default=500),
        TOP_K=_parse_int(_getenv("TOP_K"), default=5),
        SIMILARITY_THRESHOLD=_parse_float(_getenv("SIMILARITY_THRESHOLD"), default=0.0),
        MAX_FILE_SIZE=_parse_int(_getenv("MAX_FILE_SIZE"), default=50 * 1024 * 1024),
        ALLOWED_EXTENSIONS=[
            ".shp",
            ".geojson",
            ".gpkg",
            ".kml",
            ".tif",
            ".tiff",
            ".jp2",
            ".img",
            ".nc",
        ],
        PROJECT_ROOT=project_root,
        DATA_DIR=data_dir,
        PDF_DATA_DIR=data_dir / "pdfs",
        VECTOR_DATA_DIR=data_dir / "vector",
        RASTER_DATA_DIR=data_dir / "raster",
        CHROMA_PERSIST_DIR=data_dir / "chroma_db",
        UPLOAD_DIR=data_dir / "uploads",
        LOGS_DIR=project_root / "logs",
    )


settings = _build_settings()
