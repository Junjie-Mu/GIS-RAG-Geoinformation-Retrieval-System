"""
FastAPI main application
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.api.models import *
from src.rag.rag_engine import RAGEngine
from src.core.logger import log
from config.settings import settings

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="基于本地大模型的GIS地理信息RAG系统"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine instance
rag_engine = None

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    global rag_engine
    try:
        log.info("Starting GIS-RAG API service...")
        rag_engine = RAGEngine()
        log.info("GIS-RAG API service started")
    except Exception as e:
        log.error(f"启动失败: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "欢迎使用GIS-RAG API",
        "version": settings.VERSION,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload/pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file and index it"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支持PDF文件")
        
        # Validate file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="文件大小超过限制")
        
        # Save file to disk
        upload_dir = settings.PDF_DATA_DIR
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Add into RAG system
        success = rag_engine.add_pdf_document(file_path)
        
        if success:
            log.info(f"Processed PDF file: {file.filename}")
            return UploadResponse(
                filename=file.filename,
                file_size=file_size,
                file_type="pdf",
                success=True,
                message="PDF文件上传并处理成功"
            )
        else:
            raise HTTPException(status_code=500, detail="PDF文件处理失败")
            
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"上传PDF文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.post("/upload/gis", response_model=UploadResponse)
async def upload_gis_data(file: UploadFile = File(...)):
    """Upload a GIS data file and index it"""
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file_ext}")
        
        # Validate file size
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="文件大小超过限制")
        
        # Select save directory
        if file_ext in ['.shp', '.geojson', '.gpkg', '.kml']:
            save_dir = settings.VECTOR_DATA_DIR
        else:
            save_dir = settings.RASTER_DATA_DIR
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file to disk
        file_path = save_dir / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Add into RAG system
        success = rag_engine.add_gis_data(file_path)
        
        if success:
            log.info(f"Processed GIS file: {file.filename}")
            return UploadResponse(
                filename=file.filename,
                file_size=file_size,
                file_type="gis",
                success=True,
                message="GIS文件上传并处理成功"
            )
        else:
            raise HTTPException(status_code=500, detail="GIS文件处理失败")
            
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"上传GIS文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run a QA query against indexed documents"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        
        # Execute query
        result = rag_engine.query(request.question)
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search indexed documents"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="搜索查询不能为空")
        
        # Execute search
        results = rag_engine.search_documents(
            query=request.query,
            doc_type=request.doc_type
        )
        
        # Limit returned result count
        limited_results = results[:request.limit] if request.limit else results
        
        return SearchResponse(
            query=request.query,
            results=limited_results,
            total_count=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get current system information"""
    try:
        info = rag_engine.get_system_info()
        return SystemInfoResponse(**info)
        
    except Exception as e:
        log.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")

@app.delete("/system/clear")
async def clear_database():
    """Clear all indexed data from the database"""
    try:
        success = rag_engine.clear_database()
        
        if success:
            return {"message": "数据库清空成功", "success": True}
        else:
            raise HTTPException(status_code=500, detail="数据库清空失败")
            
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"清空数据库失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空数据库失败: {str(e)}")

@app.get("/files/list")
async def list_files():
    """List uploaded files"""
    try:
        files = []
        
        # List PDF files
        pdf_dir = settings.PDF_DATA_DIR
        if pdf_dir.exists():
            for pdf_file in pdf_dir.glob("*.pdf"):
                files.append({
                    "filename": pdf_file.name,
                    "file_type": "pdf",
                    "file_size": pdf_file.stat().st_size,
                    "upload_time": datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat()
                })
        
        # List vector files
        vector_dir = settings.VECTOR_DATA_DIR
        if vector_dir.exists():
            for ext in ['.shp', '.geojson', '.gpkg', '.kml']:
                for vector_file in vector_dir.glob(f"*{ext}"):
                    files.append({
                        "filename": vector_file.name,
                        "file_type": "vector",
                        "file_size": vector_file.stat().st_size,
                        "upload_time": datetime.fromtimestamp(vector_file.stat().st_mtime).isoformat()
                    })
        
        # List raster files
        raster_dir = settings.RASTER_DATA_DIR
        if raster_dir.exists():
            for ext in ['.tif', '.tiff', '.jp2', '.img', '.nc']:
                for raster_file in raster_dir.glob(f"*{ext}"):
                    files.append({
                        "filename": raster_file.name,
                        "file_type": "raster",
                        "file_size": raster_file.stat().st_size,
                        "upload_time": datetime.fromtimestamp(raster_file.stat().st_mtime).isoformat()
                    })
        
        return {"files": files, "total_count": len(files)}
        
    except Exception as e:
        log.error(f"列出文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出文件失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )





