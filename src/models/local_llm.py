"""
Local LLM model management
"""
from pathlib import Path
import torch
from typing import Optional, List, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
)
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from src.core.logger import log
from config.settings import settings

try:
    setattr(torch.classes, "__path__", [])
except Exception:
    pass

class LocalLLM(LLM):
    """Wrapper for a local LLM model"""

    # Explicitly declared Pydantic fields
    model_path: Optional[str] = Field(default=None)
    model_name: str = Field(default="qwen2-7b-instruct")
    device: str = Field(default="auto")
    max_new_tokens: int = Field(default=16384)
    is_thinking_model: bool = Field(default=False)

    # Fields excluded from serialization
    model: Any = Field(default=None, exclude=True)
    tokenizer: Any = Field(default=None, exclude=True)

    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "qwen2-7b-instruct",
        device: str = "auto",
        max_new_tokens: int = 16384,
        is_thinking_model: bool = False,
        **kwargs
    ):
        super().__init__(
            model_path=model_path,
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            is_thinking_model=is_thinking_model,
            **kwargs
        )
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            log.info(f"Loading local LLM model: {self.model_name}")
            
            model_path = self.model_path if self.model_path else self.model_name
            local_path = Path(model_path).expanduser()
            if local_path.exists():
                model_path = str(local_path)
            else:
                if local_path.is_absolute():
                    raise FileNotFoundError(f"LLM模型路径不存在: {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            
            # Set pad_token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            
            log.info("Local LLM model loaded")
            
        except Exception as e:
            log.error(f"加载本地LLM模型失败: {e}")
            raise e
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a reply using the model"""
        import time
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Record start time
            start_time = time.time()
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            tokenize_start = time.time()
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            tokenize_time = time.time() - tokenize_start
            
            input_length = len(model_inputs.input_ids[0])
            log.info(f"[DEBUG] Input tokens: {input_length}, tokenize time: {tokenize_time:.2f}s")
            
            generate_start = time.time()
            log.info(f"[DEBUG] Generation start: max_new_tokens={self.max_new_tokens}, is_thinking_model={self.is_thinking_model}")
            log.info(f"[DEBUG] Model device: {self.model.device}")
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # greedy decoding, faster
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            generate_time = time.time() - generate_start
            output_length = len(generated_ids[0]) - input_length
            log.info(
                f"[DEBUG] Generation done: output_tokens={output_length}, time={generate_time:.2f}s, speed={output_length/generate_time:.1f} tokens/s"
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            # Only handle </think> tag for thinking models (token ID 151668)
            decode_start = time.time()
            if self.is_thinking_model:
                try:
                    # Find </think> position; return content after it
                    index = len(output_ids) - output_ids[::-1].index(151668)
                    log.info(f"[DEBUG] Thinking model: found </think> position: {index}")
                except ValueError:
                    index = 0
                    log.info("[DEBUG] Thinking model: </think> not found")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n").strip()
            else:
                # Standard model: decode full output
                content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n").strip()
            
            decode_time = time.time() - decode_start
            total_time = time.time() - start_time
            log.info(f"[DEBUG] Decode time: {decode_time:.2f}s, total time: {total_time:.2f}s")
            log.info(f"[DEBUG] Output length: {len(content)} chars")
            
            return content
            
        except Exception as e:
            log.error(f"LLM生成回复失败: {e}")
            return f"抱歉，模型生成回复时出现错误: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "local_llm"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return basic model information"""
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': str(self.model.device) if self.model else None,
            'max_new_tokens': self.max_new_tokens,
            'is_thinking_model': self.is_thinking_model,
        }
    
    def generate_stream(self, prompt: str):
        """Stream model output as incremental chunks"""
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        
        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        in_thinking = False

        try:
            for new_text in streamer:
                if "<think>" in new_text and not in_thinking:
                    in_thinking = True
                    yield {"type": "thinking_start", "content": ""}
                    after = new_text.split("<think>", 1)[1] if "<think>" in new_text else ""
                    if after:
                        yield {"type": "thinking", "content": after}
                elif "</think>" in new_text and in_thinking:
                    in_thinking = False
                    before = new_text.split("</think>", 1)[0]
                    if before:
                        yield {"type": "thinking", "content": before}
                    yield {"type": "thinking_end", "content": ""}
                    after = new_text.split("</think>", 1)[1] if "</think>" in new_text else ""
                    cleaned = after.replace("<|im_end|>", "").strip()
                    if cleaned:
                        yield {"type": "answer", "content": cleaned}
                elif in_thinking:
                    yield {"type": "thinking", "content": new_text}
                else:
                    cleaned = new_text.replace("<|im_end|>", "")
                    if cleaned:
                        yield {"type": "answer", "content": cleaned}
        finally:
            thread.join()

