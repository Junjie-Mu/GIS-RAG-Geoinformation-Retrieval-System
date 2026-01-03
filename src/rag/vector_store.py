"""
Vector store management utilities.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pydantic import Field
try:
    from langchain_core.retrievers import BaseRetriever
except Exception:
    try:
        from langchain.schema import BaseRetriever
    except Exception:
        BaseRetriever = object
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.core.logger import log
from config.settings import settings

try:
    setattr(torch.classes, "__path__", [])
except Exception:
    pass


class MeanPoolingEmbeddings:
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        max_length: int = 256,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device if device in {"cuda", "cpu"} else "cpu")
        self.max_length = int(max_length)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.batch_size = int(batch_size)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(self.device)
        self.model.eval()

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        vectors: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                model_output = self.model(**encoded)
                pooled = self._mean_pooling(model_output.last_hidden_state, encoded["attention_mask"])
                if self.normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.detach().cpu().tolist())
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed_texts([text])
        return vectors[0] if vectors else []


class VectorStoreManager:
    """Manager for embedding model and Chroma vector store"""
    
    def __init__(self, collection_name: str = "gis_rag"):
        self.collection_name = collection_name
        self.persist_directory = settings.CHROMA_PERSIST_DIR
        self.embeddings = None
        self.vector_store = None
        self._bge_m3_model = None
        self._initialize_embeddings()
        self._initialize_vector_store()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            log.info("Initializing embedding model...")
            
            # Configure embedding model parameters
            model_name = settings.EMBEDDING_MODEL_NAME
            model_kwargs = {
                'device': 'cuda' if settings.DEVICE == 'cuda' else 'cpu',
                'trust_remote_code': True
            }
            encode_kwargs = {
                'normalize_embeddings': True
            }
            
            # Use local embedding model if path is configured
            if settings.EMBEDDING_MODEL_PATH and Path(settings.EMBEDDING_MODEL_PATH).exists():
                model_name = settings.EMBEDDING_MODEL_PATH
                log.info(f"Using local embedding model: {model_name}")

                if "bge-m3" in str(model_name).lower() or "bge-m3" in str(settings.EMBEDDING_MODEL_NAME).lower():
                    try:
                        from FlagEmbedding import BGEM3FlagModel
                    except Exception as e:
                        raise RuntimeError(
                            "当前配置为bge-m3混合检索，但缺少FlagEmbedding依赖，请先安装：pip install FlagEmbedding"
                        ) from e

                    self._bge_m3_model = BGEM3FlagModel(
                        model_name,
                        use_fp16=(settings.DEVICE == "cuda"),
                    )

                    self.embeddings = _BgeM3DenseEmbeddings(
                        model=self._bge_m3_model,
                        batch_size=32,
                    )
                else:
                    self.embeddings = MeanPoolingEmbeddings(
                        model_name_or_path=model_name,
                        device="cuda" if settings.DEVICE == "cuda" else "cpu",
                        max_length=settings.EMBEDDING_MAX_LENGTH,
                        normalize_embeddings=True,
                    )
                log.info("Embedding model initialized")
                return
            else:
                log.info(f"Using remote embedding model: {model_name}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            log.info("Embedding model initialized")
            
        except Exception as e:
            log.error(f"初始化嵌入模型失败: {e}")
            raise e

    def _bge_m3_encode(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        if not self._bge_m3_model:
            raise RuntimeError("bge-m3模型未初始化，无法执行混合检索")
        try:
            return self._bge_m3_model.encode(texts, **kwargs)
        except TypeError:
            kwargs.pop("max_length", None)
            return self._bge_m3_model.encode(texts, **kwargs)

    def _initialize_vector_store(self):
        """Initialize underlying Chroma vector store"""
        try:
            log.info("Initializing vector store...")
            
            # Ensure persistence directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize Chroma vector store
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            log.info(f"Vector store initialized, collection: {self.collection_name}")
            
        except Exception as e:
            log.error(f"初始化向量存储失败: {e}")
            raise e

    def as_retriever(self, k: Optional[int] = None):
        top_k = k or settings.TOP_K
        if settings.RETRIEVAL_MODE == "hybrid" and self._bge_m3_model is not None:
            return _HybridRetriever(store=self, top_k=top_k)
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Add documents into the vector store and persist them"""
        try:
            if not documents:
                log.warning("没有文档需要添加")
                return []
            
            log.info(f"Adding {len(documents)} documents to vector store")
            
            # Generate IDs if none provided
            if ids is None:
                ids = [f"doc_{i}_{hash(doc.page_content)}" for i, doc in enumerate(documents)]
            
            # Add documents
            doc_ids = self.vector_store.add_documents(documents, ids=ids)
            
            # Persist to disk
            self.vector_store.persist()
            
            log.info(f"Added {len(doc_ids)} documents to vector store")
            return doc_ids
            
        except Exception as e:
            log.error(f"添加文档到向量存储失败: {e}")
            return []
    
    def _hybrid_rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        if not docs:
            return []

        query_embeddings = self._bge_m3_encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=settings.EMBEDDING_MAX_LENGTH,
        )
        passage_embeddings = self._bge_m3_encode(
            [d.page_content for d in docs],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=settings.EMBEDDING_MAX_LENGTH,
        )

        query_dense = query_embeddings["dense_vecs"][0]
        query_lexical = query_embeddings["lexical_weights"][0]

        dense_weight = float(settings.HYBRID_DENSE_WEIGHT)
        lexical_weight = float(settings.HYBRID_LEXICAL_WEIGHT)
        weight_sum = dense_weight + lexical_weight
        if weight_sum > 0:
            dense_weight /= weight_sum
            lexical_weight /= weight_sum

        scored: List[Tuple[Document, float]] = []
        for idx, doc in enumerate(docs):
            passage_dense = passage_embeddings["dense_vecs"][idx]
            try:
                dense_sim = float(query_dense @ passage_dense.T)
            except Exception:
                dense_sim = float(
                    sum(float(a) * float(b) for a, b in zip(query_dense, passage_dense))
                )
            lexical_sim = float(
                self._bge_m3_model.compute_lexical_matching_score(
                    passage_embeddings["lexical_weights"][idx],
                    query_lexical,
                )
            )
            score = dense_weight * dense_sim + lexical_weight * lexical_sim
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def hybrid_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        top_k = k or settings.TOP_K
        candidate_k = max(top_k * max(int(settings.HYBRID_CANDIDATE_MULTIPLIER), 1), top_k)

        candidates = self.vector_store.similarity_search(
            query=query,
            k=candidate_k,
            filter=filter,
        )
        reranked = self._hybrid_rerank(query=query, docs=candidates)
        return reranked[:top_k]

    def similarity_search(
        self, 
        query: str, 
        k: int = None, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Run similarity search and return matching documents"""
        try:
            k = k or settings.TOP_K
            log.info(f"Similarity search: query={query[:50]}..., top_k={k}")

            if settings.RETRIEVAL_MODE == "hybrid" and self._bge_m3_model is not None:
                docs_with_scores = self.hybrid_search_with_score(query=query, k=k, filter=filter)
                docs = [doc for doc, _ in docs_with_scores]
            else:
                docs = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter
                )
            
            log.info(f"Found {len(docs)} relevant documents")
            return docs
            
        except Exception as e:
            log.error(f"相似度搜索失败: {e}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = None, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """Similarity search and return (doc, score) pairs"""
        try:
            k = k or settings.TOP_K
            log.info(f"Similarity search with scores: query={query[:50]}..., top_k={k}")

            if settings.RETRIEVAL_MODE == "hybrid" and self._bge_m3_model is not None:
                docs_with_scores = self.hybrid_search_with_score(query=query, k=k, filter=filter)
                log.info(f"Found {len(docs_with_scores)} relevant documents")
                return docs_with_scores

            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )

            filtered_docs = [
                (doc, score) for doc, score in docs_with_scores 
                if score >= settings.SIMILARITY_THRESHOLD
            ]

            log.info(f"Found {len(filtered_docs)} high-quality relevant documents")
            return filtered_docs
            
        except Exception as e:
            log.error(f"带分数的相似度搜索失败: {e}")
            return []
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by ID from the vector store"""
        try:
            log.info(f"Deleting {len(ids)} documents")
            
            self.vector_store.delete(ids=ids)
            self.vector_store.persist()
            
            log.info("Documents deleted")
            return True
            
        except Exception as e:
            log.error(f"删除文档失败: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Return basic collection information"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': str(self.persist_directory)
            }
            
        except Exception as e:
            log.error(f"获取集合信息失败: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Delete all vectors from the collection"""
        try:
            log.warning("清空向量存储集合")
            
            # Get all IDs and delete
            collection = self.vector_store._collection
            all_ids = collection.get()['ids']
            
            if all_ids:
                self.vector_store.delete(ids=all_ids)
                self.vector_store.persist()
            
            log.info("Vector store collection cleared")
            return True
            
        except Exception as e:
            log.error(f"清空集合失败: {e}")
            return False


class _BgeM3DenseEmbeddings:
    def __init__(self, model: Any, batch_size: int = 32):
        self._model = model
        self._batch_size = int(batch_size)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            outputs = self._model.encode(
                texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
                batch_size=self._batch_size,
            )
        except TypeError:
            outputs = self._model.encode(
                texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        return outputs["dense_vecs"].tolist() if hasattr(outputs["dense_vecs"], "tolist") else outputs["dense_vecs"]

    def embed_query(self, text: str) -> List[float]:
        vecs = self.embed_documents([text])
        return vecs[0] if vecs else []


class _HybridRetriever(BaseRetriever):
    store: Any = Field(exclude=True)
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        return self.store.similarity_search(query=query, k=self.top_k)

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)
