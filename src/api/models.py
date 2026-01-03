"""
Pydantic models for the GIS-RAG API.
"""
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request body for QA queries"""
    question: str = Field(..., description="用户问题")
    doc_type: Optional[str] = Field(None, description="文档类型过滤 (pdf, gis_metadata)")

class QueryResponse(BaseModel):
    """Response body for QA queries"""
    question: str
    answer: str
    source_documents: List[Dict[str, Any]]
    success: bool

class UploadResponse(BaseModel):
    """Response model for upload endpoints"""
    filename: str
    file_size: int
    file_type: str
    success: bool
    message: str

class SearchRequest(BaseModel):
    """Request body for document search"""
    query: str = Field(..., description="搜索查询")
    doc_type: Optional[str] = Field(None, description="文档类型")
    limit: Optional[int] = Field(10, description="返回结果数量")

class SearchResponse(BaseModel):
    """Response body for document search"""
    query: str
    results: List[Dict[str, Any]]
    total_count: int

class SystemInfoResponse(BaseModel):
    """Response model for system information"""
    vector_store: Dict[str, Any]
    llm_model: Dict[str, Any]
    processors: Dict[str, bool]
    status: str

class DocumentInfo(BaseModel):
    """Metadata model for a stored document"""
    filename: str
    file_size: int
    file_type: str
    doc_count: int
    added_at: str





