# GIS-RAG 地理信息 RAG 系统

基于本地大模型的 GIS 检索增强生成系统：支持 PDF 文档解析与 GIS（矢量/栅格）元数据提取，并提供 Web 界面与 API。

## 功能

- PDF 文档解析与索引（课件/教材/技术文档）
- 矢量/栅格数据元数据提取与检索
- 本地 LLM问答
- Web（Streamlit）与 API（FastAPI）两种入口

## 环境要求

- Python 3.10+
- 本地部署建议有8GB显存至少

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

建议在虚拟环境中运行（venv/conda 均可）。

### 2) 配置 .env（推荐）

项目会从根目录的 `.env` 读取配置（见 [settings.py](file:///d:/Desktop/GIS-RAG/config/settings.py)）。至少需要配置 Instruct 模型路径与设备：

```ini
# 基础模型
LLM_INSTRUCT_MODEL_PATH=D:\HuggingFace\Models\Qwen3-4B-Instruct-2507-FP8
LLM_INSTRUCT_MODEL_NAME=Qwen3-4B-Instruct-2507-FP8

# 可选：思考模型
LLM_THINKING_MODEL_PATH=D:\HuggingFace\Models\Qwen3-4B-Thinking-2507-FP8
LLM_THINKING_MODEL_NAME=Qwen3-4B-Thinking-2507-FP8

# 设备：cuda 或 cpu
DEVICE=cuda

# 服务端口（可选）
API_HOST=127.0.0.1
API_PORT=8000
WEB_HOST=127.0.0.1
WEB_PORT=8501
```

嵌入模型、检索策略、分块参数等也可在 `.env` 中覆盖，默认值以 [settings.py](file:///d:/Desktop/GIS-RAG/config/settings.py) 为准。

### 3) 启动 Web

```bash
python start_web.py
```

默认地址：`http://localhost:8501`

### 4) 启动 API

```bash
python start_api.py
```

默认地址：`http://localhost:8000`  
Swagger 文档：`http://localhost:8000/docs`

## 数据与目录

项目默认使用 `data/` 作为工作数据目录（见 [settings.py](file:///d:/Desktop/GIS-RAG/config/settings.py)）：

```
data/
  pdfs/          # PDF 文档
  vector/        # 矢量数据（.shp/.geojson/.gpkg/.kml）
  raster/        # 栅格数据（.tif/.tiff/.jp2/.img/.nc）
  chroma_db/     # 向量库持久化目录
  conversations/ # Web 历史对话
  uploads/       # Web 上传的原始文件缓存
logs/            # 运行日志
```

你可以直接把文件放到对应目录，也可以在 Web 页面上传。

## 支持的数据格式

- 矢量：Shapefile（.shp）、GeoJSON（.geojson）、GeoPackage（.gpkg）、KML（.kml）
- 栅格：GeoTIFF（.tif/.tiff）、JPEG2000（.jp2）、IMG（.img）、NetCDF（.nc）
- 文档：PDF（.pdf）





