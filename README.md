# GIS-RAG Geospatial RAG System

A local-LLM-powered geospatial Retrieval-Augmented Generation (RAG) system. It supports PDF parsing and GIS (vector/raster) metadata extraction, and provides both a Web UI and an API.

## Features

- PDF parsing and indexing (slides/textbooks/technical docs)
- Vector/raster GIS metadata extraction and retrieval
- Local LLM Q&A
- Two entry points: Web (Streamlit) and API (FastAPI)

## Requirements

- Python 3.10+
- For local GPU deployment, 8GB+ VRAM is recommended

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

It is recommended to run inside a virtual environment (venv/conda both work).

### 2) Configure `.env` (recommended)

The project reads configuration from `.env` in the repository root (see [settings.py](file:///d:/Desktop/GIS-RAG/config/settings.py)). At minimum, set the Instruct model path and device:

```ini
# Base model (Instruct)
LLM_INSTRUCT_MODEL_PATH=D:\HuggingFace\Models\Qwen3-4B-Instruct-2507-FP8
LLM_INSTRUCT_MODEL_NAME=Qwen3-4B-Instruct-2507-FP8

# Optional: thinking model (Deep Think)
LLM_THINKING_MODEL_PATH=D:\HuggingFace\Models\Qwen3-4B-Thinking-2507-FP8
LLM_THINKING_MODEL_NAME=Qwen3-4B-Thinking-2507-FP8

# Device: cuda or cpu
DEVICE=cuda

# Service ports (optional)
API_HOST=127.0.0.1
API_PORT=8000
WEB_HOST=127.0.0.1
WEB_PORT=8501
```

Embedding settings, retrieval strategy, chunking parameters, etc. can also be overridden via `.env`. Defaults are defined in [settings.py](file:///d:/Desktop/GIS-RAG/config/settings.py).

### 3) Start the Web UI

```bash
python start_web.py
```

Default URL: `http://localhost:8501`

### 4) Start the API

```bash
python start_api.py
```

Default URL: `http://localhost:8000`  
Swagger docs: `http://localhost:8000/docs`

## Data & Directories

By default, the project uses `data/` as its working data directory (see [settings.py](file:///d:/Desktop/GIS-RAG/config/settings.py)):

```
data/
  pdfs/          # PDF documents
  vector/        # Vector data (.shp/.geojson/.gpkg/.kml)
  raster/        # Raster data (.tif/.tiff/.jp2/.img/.nc)
  chroma_db/     # Vector store persistence (Chroma)
  conversations/ # Web chat history
  uploads/       # Raw uploaded file cache (Web)
logs/            # Runtime logs
```

You can either place files directly into these folders, or upload them via the Web UI.

## Supported Formats

- Vector: Shapefile (.shp), GeoJSON (.geojson), GeoPackage (.gpkg), KML (.kml)
- Raster: GeoTIFF (.tif/.tiff), JPEG2000 (.jp2), IMG (.img), NetCDF (.nc)
- Documents: PDF (.pdf)





