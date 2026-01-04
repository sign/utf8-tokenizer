"""
UTF-8 Grouping module.

Provides:
- group_utf8_bytes: Convert byte sequences to grouped representation (auto CUDA)
- UTF8GroupedEmbedding: Embedding layer for grouped UTF-8 bytes
- CausalLMWrapper: Wrapper to adapt CausalLM models for grouped bytes
- is_leading_byte: Check if a byte is a UTF-8 leading byte
"""

from .causal_lm import CausalLMWrapper
from .embedding import UTF8GroupedEmbedding, is_leading_byte
from .group_utf8_bytes import (
    GLOBAL_MEM_SEQ_LIMIT,
    SHARED_MEM_SEQ_LIMIT,
    group_utf8_bytes,
    is_cuda_kernel_available,
)

__all__ = [
    # Main function
    "group_utf8_bytes",
    # Embedding layer
    "UTF8GroupedEmbedding",
    # Model wrapper
    "CausalLMWrapper",
    # Utilities
    "is_leading_byte",
    "is_cuda_kernel_available",
    # Constants
    "SHARED_MEM_SEQ_LIMIT",
    "GLOBAL_MEM_SEQ_LIMIT",
]
