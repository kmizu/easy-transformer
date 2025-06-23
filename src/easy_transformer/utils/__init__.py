"""Utility functions for easy-transformer."""

from .masking import create_padding_mask, create_causal_mask, create_look_ahead_mask
from .visualization import (
    visualize_attention,
    plot_positional_encoding,
    visualize_embeddings,
    plot_loss_curves,
)
from .metrics import calculate_bleu, calculate_perplexity, calculate_accuracy
from .helpers import (
    count_parameters,
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
)
from .data import (
    create_dataloader,
    tokenize_text,
    build_vocabulary,
    pad_sequences,
)

__all__ = [
    # Masking
    "create_padding_mask",
    "create_causal_mask",
    "create_look_ahead_mask",
    # Visualization
    "visualize_attention",
    "plot_positional_encoding",
    "visualize_embeddings",
    "plot_loss_curves",
    # Metrics
    "calculate_bleu",
    "calculate_perplexity",
    "calculate_accuracy",
    # Helpers
    "count_parameters",
    "set_seed",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
    # Data
    "create_dataloader",
    "tokenize_text",
    "build_vocabulary",
    "pad_sequences",
]