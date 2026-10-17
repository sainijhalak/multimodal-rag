import asyncio
import logging
import threading
from typing import AsyncGenerator, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from ..config import settings

logger = logging.getLogger(__name__)

class LocalLLM:
    """
    Lightweight, in-process Conversational LLM using Hugging Face Transformers.
    Runs Qwen2.5-0.5B-Instruct directly on CUDA in bfloat16 (~500 MB VRAM).
    No background daemons, zero external network calls, completely offline.
    """

    def __init__(self, model_id: str = settings.LLM_MODEL, device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading local LLM ({self.model_id}) on {self.device}...")
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        logger.info("Local LLM successfully initialized.")

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256
    ) -> str:
        """Generates a complete response synchronously in a background thread."""
        def _sync_generate():
            self._load_model()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    repetition_penalty=1.15,
                    use_cache=True,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Strip input prompt tokens
            input_length = model_inputs.input_ids.shape[1]
            response_tokens = generated_ids[0][input_length:]
            return self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        return await asyncio.to_thread(_sync_generate)

    async def stream_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256
    ) -> AsyncGenerator[str, None]:
        """Streams generated tokens in real-time non-blocking using TextIteratorStreamer."""
        import queue
        self._load_model()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            repetition_penalty=1.15,
            use_cache=True,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Non-blocking async queue polling to keep the event loop and SSE response fluid
        while True:
            chunk = ""
            while not streamer.text_queue.empty():
                try:
                    tok = streamer.text_queue.get_nowait()
                    if tok == streamer.stop_signal:
                        if chunk:
                            yield chunk
                        thread.join()
                        return
                    if tok:
                        chunk += tok
                except queue.Empty:
                    break

            if chunk:
                yield chunk

            if not thread.is_alive() and streamer.text_queue.empty():
                break

            await asyncio.sleep(0.003)

        thread.join()
