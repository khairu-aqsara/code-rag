import logging
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages a single unified embedding model (gte-modernbert-base)."""

    def __init__(self, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        logger.info(f"Loading embedding model on device: {device}")

        self._tokenizer = AutoTokenizer.from_pretrained(settings.EMBED_MODEL)
        self._model = AutoModel.from_pretrained(settings.EMBED_MODEL)
        self._model.to(device)
        self._model.eval()

        # Startup validation: verify model output dimension matches config
        with torch.no_grad():
            probe = self._tokenizer("dimension check", return_tensors="pt")
            probe = {k: v.to(device) for k, v in probe.items()}
            actual_dim = self._model(**probe).last_hidden_state.shape[-1]
        if actual_dim != settings.EMBED_DIM:
            raise ValueError(
                f"Model {settings.EMBED_MODEL} outputs {actual_dim}-dim embeddings "
                f"but EMBED_DIM={settings.EMBED_DIM}. Update EMBED_DIM in .env or config."
            )

        logger.info("Model loaded successfully")

        # Compute model size in MB once at load time
        param_bytes = sum(p.numel() * p.element_size() for p in self._model.parameters())
        self._model_size_mb = param_bytes / (1024 ** 2)
        self._truncation_warned = False

    @property
    def model_name(self) -> str:
        return settings.EMBED_MODEL

    @property
    def model_size_mb(self) -> float:
        return self._model_size_mb

    def _l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of strings. Returns (N, 768) float32, L2-normalized."""
        if not texts:
            return np.empty((0, settings.EMBED_DIM), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(texts), settings.EMBED_BATCH_SIZE):
            batch = texts[i : i + settings.EMBED_BATCH_SIZE]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=settings.EMBED_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            if not self._truncation_warned and encoded["input_ids"].shape[1] >= settings.EMBED_MAX_LENGTH:
                logger.warning(f"Chunks truncated at {settings.EMBED_MAX_LENGTH} tokens — raise EMBED_MAX_LENGTH in .env if needed (max 8192)")
                self._truncation_warned = True

            with torch.no_grad():
                outputs = self._model(**encoded)

            # CLS pooling: take first token representation
            cls_embeddings = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
            all_embeddings.append(cls_embeddings)

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        return self._l2_normalize(embeddings)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (768,)."""
        return self.embed_batch([text])[0]

    # Backward-compatible aliases
    def embed_text(self, text: str) -> np.ndarray:
        """Alias for embed() — for text/document queries."""
        return self.embed(text)

    def embed_code(self, code: str) -> np.ndarray:
        """Alias for embed() — for code queries."""
        return self.embed(code)

    def embed_text_batch(self, texts: List[str]) -> np.ndarray:
        """Alias for embed_batch() — for text/document batches."""
        return self.embed_batch(texts)

    def embed_code_batch(self, snippets: List[str]) -> np.ndarray:
        """Alias for embed_batch() — for code batches."""
        return self.embed_batch(snippets)