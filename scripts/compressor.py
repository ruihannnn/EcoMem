"""
LLMLingua-2 compressor module for dialogue pre-compression.

Compresses dialogue text while preserving key information,
reducing token count for downstream memory construction.
"""
from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional


class LlmLingua2Compressor:
    """Wrapper around LLMLingua-2 PromptCompressor for dialogue pre-compression."""

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device_map: str = "cuda",
        use_llmlingua2: bool = True,
        compress_rate: float = 0.6,
        max_batch_size: int = 50,
        max_force_token: int = 100,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.use_llmlingua2 = use_llmlingua2
        self.compress_rate = compress_rate

        try:
            from llmlingua import PromptCompressor
        except ImportError:
            raise ImportError(
                "Required package 'llmlingua' not found. "
                "Please install it with: pip install llmlingua"
            )

        try:
            init_kwargs: Dict[str, Any] = {
                "model_name": model_name,
                "device_map": device_map,
            }
            if use_llmlingua2:
                init_kwargs["use_llmlingua2"] = True
                init_kwargs["llmlingua2_config"] = {
                    "max_batch_size": max_batch_size,
                    "max_force_token": max_force_token,
                }
            self._compressor = PromptCompressor(**init_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LlmLingua2Compressor: {e}")

    def compress_text(self, text: str) -> str:
        """Compress a single piece of text.

        Returns the compressed text, or the original text if compression fails.
        """
        if not text or not text.strip():
            return text

        try:
            result = self._compressor.compress_prompt(
                context=[text],
                instruction="",
                rate=self.compress_rate,
                target_token=-1,
            )
            compressed = result["compressed_prompt"]
            if compressed and compressed.strip():
                return compressed.strip()
            return text
        except Exception as e:
            err_msg = str(e) or repr(e)
            print(f"[compress_text] error, keeping original: {type(e).__name__}: {err_msg}")
            traceback.print_exc()
            return text

    def compress_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Compress the 'content'/'text' field of each message dict."""
        for msg in messages:
            for key in ("text", "content"):
                if key in msg and msg[key] and msg[key].strip():
                    msg[key] = self.compress_text(msg[key])
        return messages
