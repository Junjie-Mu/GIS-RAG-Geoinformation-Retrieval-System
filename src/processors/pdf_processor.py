"""
PDF document processor.
Supports parsing, chunking, and vectorizing PDF documents.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from src.core.logger import log
from config.settings import settings

class PDFProcessor:
    """Processor for PDF documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "!", "?", "；", ".", " ", ""]
        )
    
    def extract_text_pymupdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Get basic page image info
                image_list = page.get_images()
                
                page_data = {
                    'page_number': page_num + 1,
                    'text': text.strip(),
                    'image_count': len(image_list),
                    'metadata': {
                        'page_size': page.rect,
                        'rotation': page.rotation
                    }
                }
                pages_data.append(page_data)
            
            doc.close()
            return pages_data
            
        except Exception as e:
            log.error(f"PyMuPDF提取PDF文本失败: {e}")
            return []
    
    def extract_text_pdfplumber(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF using pdfplumber (better table support)"""
        try:
            pages_data = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text() or ""
                    
                    # Extract tables
                    tables = page.extract_tables()
                    
                    # Convert table data to plain text
                    table_text = ""
                    if tables:
                        for table in tables:
                            for row in table:
                                if row:
                                    table_text += " | ".join([cell or "" for cell in row]) + "\n"
                    
                    page_data = {
                        'page_number': page_num + 1,
                        'text': text.strip(),
                        'tables': tables,
                        'table_text': table_text.strip(),
                        'metadata': {
                            'width': page.width,
                            'height': page.height,
                            'rotation': getattr(page, 'rotation', 0)
                        }
                    }
                    pages_data.append(page_data)
            
            return pages_data
            
        except Exception as e:
            log.error(f"pdfplumber提取PDF文本失败: {e}")
            return []
    
    def process_pdf(self, pdf_path: Path, use_pdfplumber: bool = True) -> List[Document]:
        """Process a PDF and return chunked LangChain Document objects"""
        try:
            log.info(f"Processing PDF document: {pdf_path}")
            
            # Choose extraction method
            if use_pdfplumber:
                pages_data = self.extract_text_pdfplumber(pdf_path)
            else:
                pages_data = self.extract_text_pymupdf(pdf_path)
            
            if not pages_data:
                log.warning(f"未能从PDF中提取任何内容: {pdf_path}")
                return []
            
            # Merge text from all pages
            full_text = ""
            metadata = {
                'source': str(pdf_path),
                'total_pages': len(pages_data),
                'file_name': pdf_path.name,
                'file_size': pdf_path.stat().st_size,
            }
            
            for page_data in pages_data:
                page_text = page_data['text']
                if page_data.get('table_text'):
                    page_text += f"\n\nTable data:\n{page_data['table_text']}"
                
                if page_text.strip():
                    full_text += f"\n\n--- Page {page_data['page_number']} ---\n{page_text}"
            
            if not full_text.strip():
                log.warning(f"PDF文档中没有可提取的文本: {pdf_path}")
                return []
            
            # Create document object
            document = Document(
                page_content=full_text.strip(),
                metadata=metadata
            )
            
            # Split into text chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Attach extra metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_count': len(chunks),
                    'doc_type': 'pdf'
                })
            
            log.info(f"PDF processed; produced {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            log.error(f"处理PDF文档失败: {e}")
            return []
    
    def extract_images(self, pdf_path: Path, output_dir: Optional[Path] = None) -> List[str]:
        """Extract images from a PDF and save them to disk"""
        try:
            if output_dir is None:
                output_dir = settings.UPLOAD_DIR / "images"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            doc = fitz.open(pdf_path)
            image_paths = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n < 5:  # ensure RGB or grayscale image
                        img_path = output_dir / f"{pdf_path.stem}_page{page_num+1}_img{img_index+1}.png"
                        pix.save(str(img_path))
                        image_paths.append(str(img_path))
                    
                    pix = None
            
            doc.close()
            log.info(f"Extracted {len(image_paths)} images from PDF")
            return image_paths
            
        except Exception as e:
            log.error(f"提取PDF图像失败: {e}")
            return []
    
    def get_pdf_info(self, pdf_path: Path) -> Dict[str, Any]:
        """Get basic PDF information"""
        try:
            doc = fitz.open(pdf_path)
            
            info = {
                'file_name': pdf_path.name,
                'file_size': pdf_path.stat().st_size,
                'page_count': len(doc),
                'metadata': doc.metadata,
                'is_encrypted': doc.needs_pass,
                'is_pdf': doc.is_pdf,
            }
            
            doc.close()
            return info
            
        except Exception as e:
            log.error(f"获取PDF信息失败: {e}")
            return {}





