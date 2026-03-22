"""
llm_engine.py
-------------
FR7 – AI Response Generation

Loads Microsoft Phi-2 (or any causal-LM) locally via HuggingFace Transformers.
Exposes a simple generate(prompt) interface used by the RAG pipeline.

Phi-2 notes
-----------
* 2.7B parameters – runs comfortably on a modern CPU (slow) or GPU.
* Best results with the "Instruct:" / "Output:" prompt format.
* No API key required – weights are downloaded from HuggingFace Hub on first run.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton lock so the model is only loaded once even if called from multiple threads.
_load_lock = threading.Lock()
_llm_instance: Optional["Phi2LLM"] = None


class Phi2LLM:
    """
    Wrapper around the Phi-2 causal language model.

    Args:
        model_name:     HuggingFace model ID (default: microsoft/phi-2).
        max_new_tokens: Maximum tokens to generate per response.
        temperature:    Sampling temperature (0.0–1.0). Lower → more factual.
        device:         'cpu', 'cuda', or 'auto' (auto-detects GPU).
    """

    def __init__(
        self,
        model_name:     str   = "microsoft/phi-2",
        max_new_tokens: int   = 512,
        temperature:    float = 0.1,
        device:         str   = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name     = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature

        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading Phi-2 model '%s' on %s …", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Phi-2 loaded successfully.")

    def generate(self, prompt: str) -> str:
        """
        Run the model on a prompt and return the generated text (excluding the prompt).

        Args:
            prompt: Full formatted prompt string.

        Returns:
            Generated answer string.
        """
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
                temperature=max(self.temperature, 1e-6),   # avoid zero
                do_sample=self.temperature > 0.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip prompt)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer  = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return answer.strip()


# ── Prompt Templates ──────────────────────────────────────────────────────────

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
    """Format the RAG prompt for Phi-2."""
    return COMPLIANCE_PROMPT_TEMPLATE.format(context=context, question=question)


# ── Singleton accessor ────────────────────────────────────────────────────────

def get_llm(
    model_name:     str   = "microsoft/phi-2",
    max_new_tokens: int   = 512,
    temperature:    float = 0.1,
) -> "Phi2LLM":
    """
    Return the singleton Phi2LLM instance, loading it on first call.
    Thread-safe.
    """
    global _llm_instance
    if _llm_instance is None:
        with _load_lock:
            if _llm_instance is None:   # double-checked locking
                _llm_instance = Phi2LLM(
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
    return _llm_instance
