"""
GIS metadata extractor.
Supports metadata extraction for both vector and raster data.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box
from langchain.docstore.document import Document
from src.core.logger import log
from config.settings import settings

class GISMetadataExtractor:
    """Extractor for GIS vector and raster metadata"""
    
    def __init__(self):
        self.supported_vector_formats = ['.shp', '.geojson', '.gpkg', '.kml']
        self.supported_raster_formats = ['.tif', '.tiff', '.jp2', '.img', '.nc']
    
    def extract_vector_metadata(self, vector_path: Path) -> Dict[str, Any]:
        """Extract metadata from a vector dataset"""
        try:
            log.info(f"Extracting vector metadata: {vector_path}")
            
            # Read vector data
            gdf = gpd.read_file(vector_path)
            
            # Basic file information
            metadata = {
                'file_name': vector_path.name,
                'file_path': str(vector_path),
                'file_size': vector_path.stat().st_size,
                'data_type': 'vector',
                'format': vector_path.suffix.lower(),
                
                # Geometry information
                'feature_count': len(gdf),
                'geometry_type': str(gdf.geometry.geom_type.mode()[0]) if not gdf.empty else None,
                'bounds': gdf.total_bounds.tolist() if not gdf.empty else None,
                
                # Coordinate reference system information
                'crs': str(gdf.crs) if gdf.crs else None,
                'crs_name': gdf.crs.to_string() if gdf.crs else None,
                'is_geographic': gdf.crs.is_geographic if gdf.crs else None,
                'is_projected': gdf.crs.is_projected if gdf.crs else None,
                
                # Attribute information
                'columns': gdf.columns.tolist(),
                'column_types': gdf.dtypes.astype(str).to_dict(),
                'attribute_count': len(gdf.columns) - 1,  # subtract geometry column
            }
            
            # Detailed geometry type statistics
            if not gdf.empty:
                geom_types = gdf.geometry.geom_type.value_counts().to_dict()
                metadata['geometry_types'] = geom_types
                
                # Spatial extent
                bounds = gdf.total_bounds
                metadata['spatial_extent'] = {
                    'min_x': float(bounds[0]),
                    'min_y': float(bounds[1]),
                    'max_x': float(bounds[2]),
                    'max_y': float(bounds[3]),
                    'width': float(bounds[2] - bounds[0]),
                    'height': float(bounds[3] - bounds[1])
                }
            
            # Detailed attribute field information
            if len(gdf.columns) > 1:  # excluding geometry column
                attr_fields = []
                for col in gdf.columns:
                    if col != 'geometry':
                        field_info = {
                            'name': col,
                            'type': str(gdf[col].dtype),
                            'null_count': int(gdf[col].isnull().sum()),
                            'unique_count': int(gdf[col].nunique())
                        }
                        
                        # For numeric fields add statistics
                        if gdf[col].dtype in ['int64', 'float64']:
                            field_info.update({
                                'min': float(gdf[col].min()) if not gdf[col].isnull().all() else None,
                                'max': float(gdf[col].max()) if not gdf[col].isnull().all() else None,
                                'mean': float(gdf[col].mean()) if not gdf[col].isnull().all() else None
                            })
                        
                        attr_fields.append(field_info)
                
                metadata['attribute_fields'] = attr_fields
            
            log.info(f"Vector metadata extracted; feature_count={metadata['feature_count']}")
            return metadata
            
        except Exception as e:
            log.error(f"提取矢量数据元数据失败: {e}")
            return {
                'file_name': vector_path.name,
                'file_path': str(vector_path),
                'data_type': 'vector',
                'error': str(e)
            }
    
    def extract_raster_metadata(self, raster_path: Path) -> Dict[str, Any]:
        """Extract metadata from a raster dataset"""
        try:
            log.info(f"Extracting raster metadata: {raster_path}")
            
            with rasterio.open(raster_path) as src:
                # Basic file information
                metadata = {
                    'file_name': raster_path.name,
                    'file_path': str(raster_path),
                    'file_size': raster_path.stat().st_size,
                    'data_type': 'raster',
                    'format': raster_path.suffix.lower(),
                    
                    # Image properties
                    'width': src.width,
                    'height': src.height,
                    'band_count': src.count,
                    'dtype': str(src.dtypes[0]) if src.dtypes else None,
                    
                    # Coordinate reference system
                    'crs': str(src.crs) if src.crs else None,
                    'crs_name': src.crs.to_string() if src.crs else None,
                    
                    # Geotransform information
                    'transform': src.transform[:6] if src.transform else None,
                    'pixel_size_x': abs(src.transform[0]) if src.transform else None,
                    'pixel_size_y': abs(src.transform[4]) if src.transform else None,
                    
                    # Spatial bounds
                    'bounds': src.bounds[:] if src.bounds else None,
                    
                    # Nodata value
                    'nodata': src.nodata,
                    
                    # Other metadata
                    'driver': src.driver,
                    'is_tiled': src.is_tiled,
                    'compression': src.compression.value if src.compression else None,
                }
                
                # Detailed spatial extent
                if src.bounds:
                    bounds = src.bounds
                    metadata['spatial_extent'] = {
                        'min_x': float(bounds.left),
                        'min_y': float(bounds.bottom),
                        'max_x': float(bounds.right),
                        'max_y': float(bounds.top),
                        'width': float(bounds.right - bounds.left),
                        'height': float(bounds.top - bounds.bottom)
                    }
                
                # Band information
                band_info = []
                for i in range(1, src.count + 1):
                    band_meta = {
                        'band_number': i,
                        'dtype': str(src.dtypes[i-1]),
                        'nodata': src.nodatavals[i-1] if src.nodatavals else None,
                    }
                    
                    # Try to read per-band statistics
                    try:
                        stats = src.statistics(i)
                        if stats:
                            band_meta.update({
                                'min': stats.min,
                                'max': stats.max,
                                'mean': stats.mean,
                                'std': stats.std
                            })
                    except:
                        pass
                    
                    band_info.append(band_meta)
                
                metadata['bands'] = band_info
                
                # Color interpretation
                if src.colorinterp:
                    metadata['color_interpretation'] = [ci.name for ci in src.colorinterp]
                
                # Overview (pyramid) information
                metadata['overviews'] = []
                for i in range(1, src.count + 1):
                    overview_count = src.overviews(i)
                    if overview_count:
                        metadata['overviews'].append({
                            'band': i,
                            'levels': len(overview_count),
                            'factors': overview_count
                        })
            
            log.info(
                f"Raster metadata extracted; size={metadata['width']}x{metadata['height']}, bands={metadata['band_count']}"
            )
            return metadata
            
        except Exception as e:
            log.error(f"提取栅格数据元数据失败: {e}")
            return {
                'file_name': raster_path.name,
                'file_path': str(raster_path),
                'data_type': 'raster',
                'error': str(e)
            }
    
    def metadata_to_text(self, metadata: Dict[str, Any]) -> str:
        """Convert metadata dict into a readable text description"""
        try:
            text_parts = []
            
            # Basic information
            text_parts.append(f"File name: {metadata.get('file_name', 'Unknown')}")
            text_parts.append(f"Data type: {metadata.get('data_type', 'Unknown')}")
            text_parts.append(f"File format: {metadata.get('format', 'Unknown')}")
            text_parts.append(f"File size: {metadata.get('file_size', 0) / 1024 / 1024:.2f} MB")
            
            if metadata.get('error'):
                text_parts.append(f"Error: {metadata['error']}")
                return "\n".join(text_parts)
            
            # Vector-specific information
            if metadata.get('data_type') == 'vector':
                text_parts.append(f"Feature count: {metadata.get('feature_count', 0)}")
                text_parts.append(f"Geometry type: {metadata.get('geometry_type', 'Unknown')}")
                text_parts.append(f"Attribute field count: {metadata.get('attribute_count', 0)}")
                
                if metadata.get('crs'):
                    text_parts.append(f"CRS: {metadata['crs']}")
                
                if metadata.get('spatial_extent'):
                    extent = metadata['spatial_extent']
                    text_parts.append(
                        "Spatial extent: "
                        f"({extent['min_x']:.6f}, {extent['min_y']:.6f}) "
                        f"to ({extent['max_x']:.6f}, {extent['max_y']:.6f})"
                    )
                
                if metadata.get('attribute_fields'):
                    text_parts.append("Attribute fields:")
                    for field in metadata['attribute_fields']:
                        field_desc = f"  - {field['name']} ({field['type']})"
                        if field.get('min') is not None:
                            field_desc += f", range: {field['min']:.2f} - {field['max']:.2f}"
                        text_parts.append(field_desc)
            
            # Raster-specific information
            elif metadata.get('data_type') == 'raster':
                text_parts.append(
                    f"Image size: {metadata.get('width', 0)} × {metadata.get('height', 0)} pixels"
                )
                text_parts.append(f"Band count: {metadata.get('band_count', 0)}")
                text_parts.append(f"Dtype: {metadata.get('dtype', 'Unknown')}")
                
                if metadata.get('crs'):
                    text_parts.append(f"CRS: {metadata['crs']}")
                
                if metadata.get('pixel_size_x') and metadata.get('pixel_size_y'):
                    text_parts.append(
                        f"Pixel size: {metadata['pixel_size_x']:.6f} × {metadata['pixel_size_y']:.6f}"
                    )
                
                if metadata.get('spatial_extent'):
                    extent = metadata['spatial_extent']
                    text_parts.append(
                        "Spatial extent: "
                        f"({extent['min_x']:.6f}, {extent['min_y']:.6f}) "
                        f"to ({extent['max_x']:.6f}, {extent['max_y']:.6f})"
                    )
                
                if metadata.get('bands'):
                    text_parts.append("Band information:")
                    for band in metadata['bands']:
                        band_desc = f"  - Band {band['band_number']} ({band['dtype']})"
                        if band.get('min') is not None:
                            band_desc += f", value range: {band['min']:.2f} - {band['max']:.2f}"
                        text_parts.append(band_desc)
                
                if metadata.get('color_interpretation'):
                    text_parts.append(
                        f"Color interpretation: {', '.join(metadata['color_interpretation'])}"
                    )
            
            return "\n".join(text_parts)
            
        except Exception as e:
            log.error(f"元数据转文本失败: {e}")
            return f"元数据处理错误: {str(e)}"
    
    def process_gis_data(self, data_path: Path) -> List[Document]:
        """Process GIS data and return LangChain Document objects"""
        try:
            file_ext = data_path.suffix.lower()
            
            # Detect data type and extract metadata
            if file_ext in self.supported_vector_formats:
                metadata = self.extract_vector_metadata(data_path)
            elif file_ext in self.supported_raster_formats:
                metadata = self.extract_raster_metadata(data_path)
            else:
                log.warning(f"不支持的文件格式: {file_ext}")
                return []
            
            # Convert metadata to text
            text_content = self.metadata_to_text(metadata)
            
            # Filter complex metadata types (ChromaDB supports only str, int, float, bool, None)
            filtered_metadata = {}
            for key, value in metadata.items():
                if value is None:
                    filtered_metadata[key] = None
                elif isinstance(value, (str, int, float, bool)):
                    filtered_metadata[key] = value
                elif isinstance(value, (list, dict)):
                    # Convert lists and dicts to JSON strings
                    filtered_metadata[key] = json.dumps(value, ensure_ascii=False)
                else:
                    # Convert other types to string
                    filtered_metadata[key] = str(value)
            
            # Create document object
            document = Document(
                page_content=text_content,
                metadata={
                    'source': str(data_path),
                    'file_name': data_path.name,
                    'doc_type': 'gis_metadata',
                    'data_type': filtered_metadata.get('data_type', 'unknown'),
                    'format': file_ext,
                    **filtered_metadata
                }
            )
            
            log.info(f"GIS data processed: {data_path}")
            return [document]
            
        except Exception as e:
            log.error(f"处理GIS数据失败: {e}")
            return []
    
    def is_supported_format(self, file_path: Path) -> bool:
        """Check whether file format is supported"""
        ext = file_path.suffix.lower()
        return ext in (self.supported_vector_formats + self.supported_raster_formats)





