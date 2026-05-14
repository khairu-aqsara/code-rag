"""Unit tests for EmbeddingService using mocked models."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import src.embedder


class TestEmbeddingService:
    """Tests EmbeddingService with mocked HF models (no model downloads)."""

    @patch("src.embedder.AutoModel.from_pretrained")
    @patch("src.embedder.AutoTokenizer.from_pretrained")
    def test_embed_batch_shape(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        from src.config import settings

        mock_tokenizer = MagicMock()
        encoded = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_tokenizer.side_effect = lambda *a, **kw: encoded
        mock_tokenizer_cls.return_value = mock_tokenizer

        mock_model = MagicMock()
        hidden = torch.rand(1, 10, settings.EMBED_DIM)
        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model.return_value = mock_output
        mock_model_cls.return_value = mock_model

        from src.embedder import EmbeddingService
        svc = EmbeddingService(device="cpu")
        result = svc.embed_batch(["hello world"])
        assert result.shape == (1, settings.EMBED_DIM)
        assert result.dtype == np.float32

    @patch("src.embedder.AutoModel.from_pretrained")
    @patch("src.embedder.AutoTokenizer.from_pretrained")
    def test_l2_normalization(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        from src.config import settings

        encoded = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        mock_tokenizer_cls.return_value = MagicMock(side_effect=lambda *a, **kw: encoded)
        hidden = torch.rand(1, 5, settings.EMBED_DIM) * 10
        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model_cls.return_value = MagicMock(return_value=mock_output)

        from src.embedder import EmbeddingService
        svc = EmbeddingService(device="cpu")
        result = svc.embed_batch(["test"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    @patch("src.embedder.AutoModel.from_pretrained")
    @patch("src.embedder.AutoTokenizer.from_pretrained")
    def test_embed_single_wrapper(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        from src.config import settings

        encoded = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        mock_tokenizer_cls.return_value = MagicMock(side_effect=lambda *a, **kw: encoded)
        hidden = torch.rand(1, 5, settings.EMBED_DIM)
        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model_cls.return_value = MagicMock(return_value=mock_output)

        from src.embedder import EmbeddingService
        svc = EmbeddingService(device="cpu")
        result = svc.embed("single text")
        assert result.shape == (settings.EMBED_DIM,)

    @patch("src.embedder.AutoModel.from_pretrained")
    @patch("src.embedder.AutoTokenizer.from_pretrained")
    def test_cls_pooling(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        from src.config import settings

        encoded = {
            "input_ids": torch.ones(1, 8, dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }
        mock_tokenizer_cls.return_value = MagicMock(side_effect=lambda *a, **kw: encoded)
        hidden = torch.rand(1, 8, settings.EMBED_DIM)
        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model_cls.return_value = MagicMock(return_value=mock_output)

        from src.embedder import EmbeddingService
        svc = EmbeddingService(device="cpu")
        result = svc.embed("test cls pooling")
        assert result.shape == (settings.EMBED_DIM,)

    @patch("src.embedder.AutoModel.from_pretrained")
    @patch("src.embedder.AutoTokenizer.from_pretrained")
    def test_empty_batch_returns_empty(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        from src.config import settings

        encoded = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        mock_tokenizer_cls.return_value = MagicMock(side_effect=lambda *a, **kw: encoded)
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(1, 5, settings.EMBED_DIM)
        mock_model_cls.return_value = MagicMock(return_value=mock_output)

        from src.embedder import EmbeddingService
        svc = EmbeddingService(device="cpu")
        result = svc.embed_batch([])
        assert result.shape == (0, settings.EMBED_DIM)

        result2 = svc.embed_text_batch([])
        assert result2.shape == (0, settings.EMBED_DIM)

        result3 = svc.embed_code_batch([])
        assert result3.shape == (0, settings.EMBED_DIM)

    @patch("src.embedder.AutoModel.from_pretrained")
    @patch("src.embedder.AutoTokenizer.from_pretrained")
    def test_backward_compat_aliases(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        from src.config import settings

        encoded = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        mock_tokenizer_cls.return_value = MagicMock(side_effect=lambda *a, **kw: encoded)
        hidden = torch.rand(1, 5, settings.EMBED_DIM)
        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model_cls.return_value = MagicMock(return_value=mock_output)

        from src.embedder import EmbeddingService
        svc = EmbeddingService(device="cpu")

        single = svc.embed("test")
        text = svc.embed_text("test")
        code = svc.embed_code("test")
        assert single.shape == text.shape == code.shape == (settings.EMBED_DIM,)

    @patch("src.embedder.AutoModel.from_pretrained")
    @patch("src.embedder.AutoTokenizer.from_pretrained")
    def test_dimension_validation_raises_on_mismatch(self, mock_tokenizer_cls, mock_model_cls):
        import torch

        encoded = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        mock_tokenizer_cls.return_value = MagicMock(side_effect=lambda *a, **kw: encoded)
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(1, 5, 512)
        mock_model_cls.return_value = MagicMock(return_value=mock_output)

        from src.embedder import EmbeddingService
        with pytest.raises(ValueError, match="outputs 512-dim"):
            EmbeddingService(device="cpu")

    @patch("src.embedder.AutoModel.from_pretrained")
    @patch("src.embedder.AutoTokenizer.from_pretrained")
    def test_embed_text_code_produce_same_vector(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        from src.config import settings

        encoded = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        mock_tokenizer_cls.return_value = MagicMock(side_effect=lambda *a, **kw: encoded)
        hidden = torch.rand(1, 5, settings.EMBED_DIM)
        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model_cls.return_value = MagicMock(return_value=mock_output)

        from src.embedder import EmbeddingService
        svc = EmbeddingService(device="cpu")
        text_vec = svc.embed_text("hello")
        code_vec = svc.embed_code("hello")
        np.testing.assert_array_equal(text_vec, code_vec)