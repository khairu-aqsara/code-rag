import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import settings

logger = logging.getLogger(__name__)

SUMMARIZE_MODEL = "Salesforce/codet5-small"
SUMMARIZE_MAX_INPUT = 512
SUMMARIZE_MAX_OUTPUT = 64
SUMMARIZE_MIN_CONTENT = 50


class CodeSummarizer:
    """Loaded only during ingestion — not part of the API server."""

    def __init__(self):
        logger.info(f"Loading summarizer: {SUMMARIZE_MODEL}")
        self._tok = AutoTokenizer.from_pretrained(SUMMARIZE_MODEL)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZE_MODEL)
        self._model.eval()

    def summarize(self, code: str, fallback: str = "") -> str:
        """Return a 1-2 sentence natural language description of the code."""
        if len(code.strip()) < SUMMARIZE_MIN_CONTENT:
            return fallback
        try:
            inputs = self._tok(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=SUMMARIZE_MAX_INPUT,
            )
            with torch.no_grad():
                ids = self._model.generate(
                    inputs["input_ids"],
                    max_new_tokens=SUMMARIZE_MAX_OUTPUT,
                    num_beams=1,
                )
            return self._tok.decode(ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Summarization failed, using fallback: {e}")
            return fallback

    def summarize_batch(self, codes: list[str], fallbacks: list[str]) -> list[str]:
        """Process multiple code snippets for summarization.

        Uses batch tokenization to pre-encode all inputs upfront,
        then sequential generation with greedy decoding.
        """
        tokenized = self._tok(
            codes,
            padding=True,
            truncation=True,
            max_length=SUMMARIZE_MAX_INPUT,
            return_tensors="pt",
        )
        results: list[str] = []
        for i, (code, fallback) in enumerate(zip(codes, fallbacks)):
            if len(code.strip()) < SUMMARIZE_MIN_CONTENT:
                results.append(fallback)
                continue
            try:
                input_ids = tokenized["input_ids"][i].unsqueeze(0)
                attention_mask = tokenized["attention_mask"][i].unsqueeze(0)
                with torch.no_grad():
                    ids = self._model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=SUMMARIZE_MAX_OUTPUT,
                        num_beams=1,
                    )
                results.append(self._tok.decode(ids[0], skip_special_tokens=True))
            except Exception as e:
                logger.warning(f"Summarization failed for chunk {i}, using fallback: {e}")
                results.append(fallback)
        return results