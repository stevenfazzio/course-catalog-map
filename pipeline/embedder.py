"""The one embedding model used everywhere.

Stage 02 embeds the documents with it; stage 04 hands the same instance to Toponymy so keyphrases
and exemplars land in the same space, with the same prompt, as the documents they are compared to.
Which model is `config.EMBED_MODELS[key]`; the map uses `config.EMBED_MODEL_KEY`.
"""

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import config


def compose_embed_text(courses: pd.DataFrame) -> pd.Series:
    """Title + description. Department and school stay out (see CLAUDE.md). Title alone if no description."""
    title = courses["title"].str.strip()
    desc = courses["description"].str.strip()
    return title.where(desc == "", title + "\n" + desc)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CourseEmbedder:
    """Satisfies toponymy.embedding_wrappers.TextEmbedderProtocol: encode(texts, show_progress_bar, ...)."""

    def __init__(self, key: str | None = None, device: str | None = None):
        self.key = key or config.EMBED_MODEL_KEY
        self.spec = config.EMBED_MODELS[self.key]
        self.device = device or pick_device()
        self.model = SentenceTransformer(
            self.spec["model"],
            revision=self.spec["revision"],
            device=self.device,
            trust_remote_code=self.spec["trust_remote_code"],
        )
        self.model.max_seq_length = self.spec["max_seq_length"]

    def encode(self, texts, show_progress_bar: bool | None = False, *args, **kwargs) -> np.ndarray:
        return self.model.encode(
            list(texts),
            prompt=self.spec["prompt"] or None,
            batch_size=kwargs.pop("batch_size", config.EMBED_BATCH_SIZE),
            show_progress_bar=bool(show_progress_bar),
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def describe(self) -> dict:
        import sentence_transformers

        return {
            "key": self.key,
            "model": self.spec["model"],
            "revision": self.spec["revision"],
            "family": self.spec["family"],
            "prompt": self.spec["prompt"],
            "max_seq_length": self.spec["max_seq_length"],
            "normalize_embeddings": True,
            "dtype": str(next(self.model.parameters()).dtype),
            "device": self.device,
            "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
        }
