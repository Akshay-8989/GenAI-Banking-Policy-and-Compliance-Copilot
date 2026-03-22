"""
llm_phi2.py
-----------
Local Phi-2 inference wrapper using HuggingFace Transformers.
Phi-2 is a 2.7B parameter SLM by Microsoft that runs on CPU.
Covers BRD FR7 – AI Response Generation.
"""

import os
import re
from typing import List, Optional, Iterator

from loguru import logger


# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are a Banking Compliance Assistant. Your job is to answer questions
about banking policies using ONLY the context provided below. 
Do NOT make up information. If the answer is not in the context, say so clearly.

Context from policy documents:
{context}

Question: {question}

Instructions:
- Answer strictly based on the context above.
- Be precise and professional.
- If quoting a policy, mention which document it comes from.
- Keep your answer concise but complete.

Answer:"""


SYSTEM_PROMPT = (
    "You are a Banking Compliance Assistant specialised in KYC, AML, and risk policies. "
    "Answer only using the provided document context. Never hallucinate policy details."
)


# ──────────────────────────────────────────────
# Phi-2 LLM Wrapper
# ──────────────────────────────────────────────

class Phi2LLM:
    """
    Wraps microsoft/phi-2 for local inference.
    Model is lazy-loaded on first call.
    Falls back to a rule-based stub if torch is unavailable (for testing).
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device

        self._model = None
        self._tokenizer = None
        self._loaded = False

        logger.info(f"Phi2LLM configured: {model_name} on {device}")

    def _load(self):
        """Lazy-load Phi-2 model and tokenizer."""
        if self._loaded:
            return

        logger.info("Loading Phi-2 model… (this may take 2-4 minutes on first run)")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,   # use float32 for CPU stability
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._model.eval()

            if self.device != "cpu":
                self._model = self._model.to(self.device)

            self._loaded = True
            logger.success("Phi-2 model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load Phi-2: {e}")
            logger.warning("Falling back to stub LLM for testing.")
            self._loaded = True   # prevent retry loop
            self._model = None

    def build_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Assemble the RAG prompt from retrieved context chunks."""
        context = "\n\n---\n\n".join(
            f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        return RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(self, question: str, context_chunks: List[str]) -> str:
        """
        Generate an answer given a question and list of context strings.
        Returns the answer as a plain string.
        """
        self._load()

        prompt = self.build_prompt(question, context_chunks)

        if self._model is None:
            # Stub fallback for environments without GPU/enough RAM
            return self._stub_answer(question, context_chunks)

        try:
            import torch

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=(self.temperature > 0),
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Decode only the newly generated tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

            return self._post_process(answer)

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating answer: {e}"

    def _post_process(self, text: str) -> str:
        """Clean up model output."""
        # Remove any trailing prompt leakage
        for stop in ["Question:", "Context:", "Instructions:", "Human:", "User:"]:
            if stop in text:
                text = text[:text.index(stop)]
        return text.strip()

    def _stub_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Minimal fallback: returns the most relevant sentence from context.
        Used when Phi-2 cannot be loaded (low-RAM CI/test environments).
        """
        logger.warning("Using stub LLM answer (Phi-2 not loaded).")
        if not context_chunks:
            return "No relevant policy information found for this question."

        # Return first 3 sentences of the top chunk
        top_chunk = context_chunks[0]
        sentences = re.split(r"(?<=[.!?])\s+", top_chunk)
        summary = " ".join(sentences[:3])
        return f"Based on policy documents:\n\n{summary}\n\n[Note: This is a simplified response. Full Phi-2 model not loaded.]"
