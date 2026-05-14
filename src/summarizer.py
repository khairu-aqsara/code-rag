import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

SUMMARIZE_MODEL = "Salesforce/codet5-small"
SUMMARIZE_MAX_INPUT = 512
SUMMARIZE_MAX_OUTPUT = 64
SUMMARIZE_MIN_CONTENT = 50


class CodeSummarizer:
    """Loaded only during ingestion — not part of the API server."""

    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"Loading summarizer: {SUMMARIZE_MODEL} on {self.device}")
        self._tok = AutoTokenizer.from_pretrained(SUMMARIZE_MODEL)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZE_MODEL)
        self._model.to(self.device)
        self._model.eval()

    def summarize(self, code: str, fallback: str = "") -> str:
        """Return a 1-2 sentence natural language description of the code."""
        results = self.summarize_batch([code], [fallback])
        return results[0]

    def summarize_batch(self, codes: list[str], fallbacks: list[str]) -> list[str]:
        """Summarize a batch of code snippets on the auto-detected device."""
        results: list[str] = []
        pending_indices: list[int] = []
        pending_codes: list[str] = []

        # Pass-through chunks that are too short to summarise
        for i, (code, fallback) in enumerate(zip(codes, fallbacks)):
            if len(code.strip()) < SUMMARIZE_MIN_CONTENT:
                results.append(fallback)
            else:
                results.append("")  # placeholder
                pending_indices.append(i)
                pending_codes.append(code)

        if not pending_codes:
            return results

        try:
            tokenized = self._tok(
                pending_codes,
                padding=True,
                truncation=True,
                max_length=SUMMARIZE_MAX_INPUT,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=SUMMARIZE_MAX_OUTPUT,
                    num_beams=1,
                )

            for i, (orig_idx, out) in enumerate(zip(pending_indices, output_ids)):
                results[orig_idx] = self._tok.decode(out, skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Batch summarization failed, using fallbacks: {e}")
            for orig_idx, code_idx in enumerate(pending_indices):
                results[code_idx] = fallbacks[code_idx]

        return results