"""
Script to launch the GIS-RAG API service.
"""
import uvicorn
from config.settings import settings

if __name__ == "__main__":
    show_host = settings.API_HOST
    if show_host in {"0.0.0.0", "::"}:
        show_host = "127.0.0.1"

    print(f"🚀 启动GIS-RAG API服务...")
    print(f"📡 地址: http://{show_host}:{settings.API_PORT}")
    print(f"📖 API文档: http://{show_host}:{settings.API_PORT}/docs")
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )



