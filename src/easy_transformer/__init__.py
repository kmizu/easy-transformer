"""Easy Transformer - A comprehensive tutorial on Transformers for programming language implementers."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import (
    Transformer,
    TransformerEncoder,
    TransformerDecoder,
    MultiHeadAttention,
    PositionalEncoding,
    TokenEmbedding,
)

from .utils import (
    create_padding_mask,
    create_causal_mask,
    visualize_attention,
    count_parameters,
)

__all__ = [
    # Models
    "Transformer",
    "TransformerEncoder", 
    "TransformerDecoder",
    "MultiHeadAttention",
    "PositionalEncoding",
    "TokenEmbedding",
    # Utils
    "create_padding_mask",
    "create_causal_mask",
    "visualize_attention",
    "count_parameters",
]