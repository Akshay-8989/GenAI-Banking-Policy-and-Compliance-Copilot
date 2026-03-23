"""
llm_engine.py
-------------
FR7 – AI Response Generation
Loads Phi-2 from a LOCAL folder — zero internet, zero HuggingFace calls.
"""

from __future__ import annotations
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_load_lock    = threading.Lock()
_llm_instance: Optional["Phi2LLM"] = None


class Phi2LLM:
    """
    Loads Microsoft Phi-2 from a local folder on disk.
    Pass the full local path, e.g.  r"D:\phi-2"
    """

    def __init__(
        self,
        model_path:     str,
        max_new_tokens: int   = 256,
        temperature:    float = 0.1,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path     = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading Phi-2 from LOCAL path: %s  on  %s", model_path, self.device)

        # local_files_only=True  → never touch the internet
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        logger.info("Phi-2 loaded successfully from local folder.")

    def generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=max(self.temperature, 1e-6),
                do_sample=self.temperature > 0.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip prompt)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── Prompt Template ───────────────────────────────────────────────────────────

COMPLIANCE_PROMPT_TEMPLATE = """\
Instruct: You are a precise Banking Policy & Compliance AI assistant.
Answer the user's question using ONLY the provided policy document context.
If the answer is not found in the context, say "I could not find this information in the provided policy documents."
Do NOT make up information. Be concise and cite the relevant policy section when possible.

Context from policy documents:
{context}

Question: {question}

Output:\
"""


def build_prompt(context: str, question: str) -> str:
    return COMPLIANCE_PROMPT_TEMPLATE.format(context=context, question=question)


# ── Singleton accessor ────────────────────────────────────────────────────────

def get_llm(
    model_name:     str   = None,
    max_new_tokens: int   = 256,
    temperature:    float = 0.1,
) -> "Phi2LLM":
    """
    Returns the singleton Phi2LLM instance.
    model_name must be the full local folder path, e.g. r"D:\phi-2"
    """
    global _llm_instance
    if _llm_instance is None:
        with _load_lock:
            if _llm_instance is None:
                if not model_name:
                    raise ValueError(
                        "model_name (local path to Phi-2 folder) must be set in config.py. "
                        "Example:  PHI2_MODEL_NAME = r'D:\\phi-2'"
                    )
                _llm_instance = Phi2LLM(
                    model_path=model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
    return _llm_instance
