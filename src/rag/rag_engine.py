"""
RAG engine core
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.core.logger import log
from src.models.local_llm import LocalLLM
from src.rag.vector_store import VectorStoreManager
from src.processors.pdf_processor import PDFProcessor
from src.extractors.gis_metadata_extractor import GISMetadataExtractor
from config.settings import settings

class RAGEngine:
    """Main RAG engine class"""
    
    def __init__(self, think_mode: bool = False):
        # Initialize components
        self.llm = None
        self.vector_store = None
        self.pdf_processor = PDFProcessor()
        self.gis_extractor = GISMetadataExtractor()
        self.qa_chain = None
        self._think_mode = think_mode  # Think mode flag
        
        # Initialize system
        self._initialize_components()
    
    @property
    def think_mode(self) -> bool:
        """Get current thinking mode state"""
        return self._think_mode
    
    def switch_model(self, think_mode: bool) -> bool:
        """Switch between thinking and standard models.

        Args:
            think_mode: True for thinking model, False for standard model.

        Returns:
            bool: True if the switch succeeded.
        """
        if think_mode == self._think_mode:
            log.info(f"Model already in {'thinking' if think_mode else 'standard'} mode; no switch needed")
            return True
        
        try:
            log.info(f"Switching to {'thinking' if think_mode else 'standard'} mode...")
            
            # Release current model
            if self.llm:
                del self.llm
                self.llm = None
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Load new model
            self._think_mode = think_mode
            self._load_llm()
            
            # Re-initialize QA chain
            self._initialize_qa_chain()
            
            log.info(f"Model switched: {'thinking' if think_mode else 'standard'} mode")
            return True
            
        except Exception as e:
            log.error(f"模型切换失败: {e}")
            return False
    
    def _get_model_config(self) -> tuple:
        """Get model path and name based on thinking mode"""
        if self._think_mode:
            return settings.LLM_THINKING_MODEL_PATH, settings.LLM_THINKING_MODEL_NAME
        else:
            return settings.LLM_INSTRUCT_MODEL_PATH, settings.LLM_INSTRUCT_MODEL_NAME
    
    def _load_llm(self):
        """Load LLM model"""
        model_path, model_name = self._get_model_config()
        
        if model_path:
            llm_path = Path(model_path).expanduser()
            if llm_path.exists():
                try:
                    self.llm = LocalLLM(
                        model_path=str(llm_path),
                        model_name=model_name,
                        device=settings.DEVICE,
                        is_thinking_model=self._think_mode
                    )
                    log.info(f"Local LLM initialized: {model_name}")
                except Exception as e:
                    log.error(f"本地LLM初始化失败，将降级为仅检索模式: {e}")
                    self.llm = None
            else:
                log.warning(f"LLM模型路径不存在，将降级为仅检索模式: {model_path}")
        else:
            log.warning("未配置本地LLM路径，将降级为仅检索模式")
    
    def _initialize_components(self):
        """Initialize all core components"""
        try:
            log.info("Initializing RAG engine components...")
            
            # Initialize vector store
            self.vector_store = VectorStoreManager()
            log.info("Vector store initialized")
            
            # Initialize local LLM
            self._load_llm()
            
            # Initialize QA chain
            self._initialize_qa_chain()
            
            log.info("RAG engine initialized")
            
        except Exception as e:
            log.error(f"RAG引擎初始化失败: {e}")
            raise e
    
    def _initialize_qa_chain(self):
        """Initialize Retrieval-QA chain"""
        try:
            if not self.llm:
                log.warning("LLM未初始化，跳过QA链初始化")
                return
            
            # Define prompt template
            template = """You are a professional GIS assistant. Answer the user's question using ONLY the relevant information provided.

Relevant information:
{context}

User question: {question}

Provide a clear, accurate, and detailed answer. If the information is insufficient, explicitly say what additional information is needed.

Answer:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create retriever
            retriever = self.vector_store.as_retriever(k=settings.TOP_K)
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            log.info("QA chain initialized")
            
        except Exception as e:
            log.error(f"初始化QA链失败: {e}")
    
    def add_pdf_document(self, pdf_path: Union[str, Path]) -> bool:
        """Add a PDF document into the vector store"""
        try:
            pdf_path = Path(pdf_path)
            log.info(f"Adding PDF document: {pdf_path}")
            
            # Process PDF into document chunks
            documents = self.pdf_processor.process_pdf(pdf_path)
            
            if not documents:
                log.warning(f"PDF文档处理失败或为空: {pdf_path}")
                return False
            
            # Add documents to vector store
            doc_ids = self.vector_store.add_documents(documents)
            
            if doc_ids:
                log.info(f"Added PDF document; created {len(doc_ids)} chunks")
                return True
            else:
                log.error("文档添加到向量存储失败")
                return False
                
        except Exception as e:
            log.error(f"添加PDF文档失败: {e}")
            return False
    
    def add_gis_data(self, data_path: Union[str, Path]) -> bool:
        """Add GIS data metadata into the vector store"""
        try:
            data_path = Path(data_path)
            log.info(f"Adding GIS data: {data_path}")
            
            # Validate file format
            if not self.gis_extractor.is_supported_format(data_path):
                log.warning(f"不支持的GIS数据格式: {data_path.suffix}")
                return False
            
            # Extract GIS metadata
            documents = self.gis_extractor.process_gis_data(data_path)
            
            if not documents:
                log.warning(f"GIS数据处理失败或为空: {data_path}")
                return False
            
            # Add documents to vector store
            doc_ids = self.vector_store.add_documents(documents)
            
            if doc_ids:
                log.info("Added GIS metadata")
                return True
            else:
                log.error("GIS数据添加到向量存储失败")
                return False
                
        except Exception as e:
            log.error(f"添加GIS数据失败: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Run a standard (non-streaming) QA query"""
        try:
            log.info(f"Handling query: {question}")
            
            if not self.qa_chain:
                # If QA chain is unavailable, fall back to simple retrieval
                return self._simple_retrieval(question)
            
            # Use QA chain to answer
            if hasattr(self.qa_chain, "invoke"):
                result = self.qa_chain.invoke({"query": question})
            else:
                result = self.qa_chain({"query": question})
            
            # Format response
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ],
                "success": True
            }
            
            log.info("Query handled")
            return response
            
        except Exception as e:
            log.error(f"查询处理失败: {e}")
            return {
                "question": question,
                "answer": f"查询处理失败: {str(e)}",
                "source_documents": [],
                "success": False
            }
    
    def query_stream(self, question: str, file_filter: str = None):
        """Stream QA result as a generator.

        Args:
            question: User question.
            file_filter: Optional file path to restrict retrieval; None searches all.

        Yields:
            dict: {"type": "thinking"/"answer"/"source"/"error", "content": ...}
        """
        try:
            log.info(f"Streaming query: {question}, file_filter: {file_filter}")
            
            if not self.llm:
                yield {"type": "error", "content": "LLM未初始化"}
                return
            
            # First retrieve related documents
            if file_filter:
                # Retrieve based on specific file
                filter_dict = {"source": file_filter}
                docs = self.vector_store.similarity_search(
                    question, k=settings.TOP_K, filter=filter_dict
                )
                if not docs:
                    # If nothing is found, try filtering by file name
                    from pathlib import Path
                    file_name = Path(file_filter).name
                    filter_dict = {"file_name": file_name}
                    docs = self.vector_store.similarity_search(
                        question, k=settings.TOP_K, filter=filter_dict
                    )
            else:
                docs = self.vector_store.similarity_search(question, k=settings.TOP_K)
            
            # Build context string
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Build full prompt
            if file_filter:
                from pathlib import Path
                file_name = Path(file_filter).name
                prompt = f"""You are a professional GIS assistant. The user is asking about the file "{file_name}".

File content / metadata:
{context}

User question: {question}

Provide a clear, accurate, and detailed answer grounded in the file information above. If the information is insufficient, explicitly say what additional information is needed.

Answer:"""
            else:
                prompt = f"""You are a professional GIS assistant. Answer the user's question using ONLY the relevant information provided.

Relevant information:
{context}

User question: {question}

Provide a clear, accurate, and detailed answer. If the information is insufficient, explicitly say what additional information is needed.

Answer:"""
            
            # Return source document summaries
            yield {
                "type": "source",
                "content": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in docs
                ]
            }
            
            # Stream generated answer chunks
            for chunk in self.llm.generate_stream(prompt):
                yield chunk
                
        except Exception as e:
            log.error(f"流式查询失败: {e}")
            yield {"type": "error", "content": str(e)}
    
    def _simple_retrieval(self, question: str) -> Dict[str, Any]:
        """Simple retrieval-only answer (used when LLM is unavailable)"""
        try:
            # Run similarity search
            docs = self.vector_store.similarity_search(question, k=settings.TOP_K)
            
            if not docs:
                return {
                    "question": question,
                    "answer": "抱歉，没有找到相关信息。",
                    "source_documents": [],
                    "success": True
                }
            
            # Combine retrieved content
            context = "\n\n".join([doc.page_content for doc in docs])
            
            response = {
                "question": question,
                "answer": f"根据检索到的信息:\n\n{context}",
                "source_documents": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in docs
                ],
                "success": True
            }
            
            return response
            
        except Exception as e:
            log.error(f"简单检索失败: {e}")
            return {
                "question": question,
                "answer": f"检索失败: {str(e)}",
                "source_documents": [],
                "success": False
            }
    
    def search_documents(self, query: str, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search documents and return scored chunks"""
        try:
            log.info(f"Searching documents: {query}, type: {doc_type}")
            
            # Build metadata filters
            filter_dict = {}
            if doc_type:
                filter_dict["doc_type"] = doc_type
            
            # Execute similarity search
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=settings.TOP_K * 2,  # fetch more results for filtering
                filter=filter_dict if filter_dict else None
            )
            
            # Format results
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            log.info(f"Found {len(results)} related documents")
            return results
            
        except Exception as e:
            log.error(f"搜索文档失败: {e}")
            return []
    
    def get_system_info(self) -> Dict[str, Any]:
        """Return current system status and configuration"""
        try:
            info = {
                "vector_store": self.vector_store.get_collection_info() if self.vector_store else {},
                "llm_model": self.llm.get_model_info() if self.llm else {},
                "think_mode": self._think_mode,
                "processors": {
                    "pdf_processor": True,
                    "gis_extractor": True
                },
                "status": "ready" if self.qa_chain else "limited"
            }
            
            return info
            
        except Exception as e:
            log.error(f"获取系统信息失败: {e}")
            return {}
    
    def clear_database(self) -> bool:
        """Clear all vectors from the database"""
        try:
            log.warning("清空向量数据库")
            return self.vector_store.clear_collection()
            
        except Exception as e:
            log.error(f"清空数据库失败: {e}")
            return False
