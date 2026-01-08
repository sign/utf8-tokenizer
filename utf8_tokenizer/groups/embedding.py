"""
UTF-8 Grouping Embedding Layer.

Provides encode/decode between byte indices and grouped embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from utf8_tokenizer.byte_embeddings import PatchedBitEmbeddings


def is_leading_byte(byte: int) -> bool:
    """Check if a byte is a UTF-8 leading byte (not a continuation byte)."""
    # Continuation bytes are 10xxxxxx (128-191)
    # Leading bytes are 0xxxxxxx (0-127) or 11xxxxxx (192-255)
    return byte < 128 or byte >= 192


class UTF8GroupedEmbedding(nn.Module):
    """
    UTF-8 Grouped Embedding layer for UTF-8 byte sequences.

    Encodes groups of 1-4 bytes into a fixed-size embedding and can decode back
    to the original byte indices using the embedding transpose.

    Supports batched sequences: (batch, seq, 4) -> (batch, seq, embedding_size)

    Args:
        embedding_size: Total dimension of the grouped output (must be divisible by 4)
    """

    def __init__(self, embedding_size: int = 256):
        super().__init__()

        if embedding_size % 4 != 0:
            msg = f"embedding_size must be divisible by 4, got {embedding_size}"
            raise ValueError(msg)

        self.embedding_size = embedding_size
        self.byte_dim = embedding_size // 4  # Alias for compatibility
        self.byte_embedding_dim = self.byte_dim
        self.num_embeddings = 256  # All possible byte values
        self.max_bytes = 4

        # Learnable embedding matrix, initialized with normalized rows for roundtrip
        self.embedding = nn.Embedding(self.num_embeddings, self.byte_dim)

        with torch.no_grad():
            self.embedding.weight.copy_(functional.normalize(self.embedding.weight, p=2, dim=1))

        # Patch embedding layer to support bit-biasing
        self.embedding = PatchedBitEmbeddings(self.embedding)

    def encode(self, byte_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode byte indices to grouped embeddings.

        Args:
            byte_indices: Tensor of byte values (0-255)
                - Shape (batch, 4) -> returns (batch, embedding_size)
                - Shape (batch, seq, 4) -> returns (batch, seq, embedding_size)

        Returns:
            Grouped embeddings
        """
        input_shape = byte_indices.shape

        if len(input_shape) == 2:
            # (batch, 4) -> (batch, embedding_size)
            embeddings = self.embedding(byte_indices)  # (batch, 4, byte_dim)
            return embeddings.view(input_shape[0], -1)

        elif len(input_shape) == 3:
            # (batch, seq, 4) -> (batch, seq, embedding_size)
            batch, seq, _ = input_shape
            embeddings = self.embedding(byte_indices)  # (batch, seq, 4, byte_dim)
            return embeddings.view(batch, seq, -1)

        msg = f"Expected 2D or 3D input, got {len(input_shape)}D"
        raise ValueError(msg)

    def decode(
        self, grouped: torch.Tensor, compute_decoded: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Decode grouped representation back to byte indices.

        Args:
            grouped: Grouped representation
                - Shape (batch, embedding_size) -> returns (batch, 4)
                - Shape (batch, seq, embedding_size) -> returns (batch, seq, 4)
            compute_decoded: If True, compute argmax to get decoded bytes.
                If False, return None for decoded (faster for training).

        Returns:
            Tuple of (decoded_bytes, logits). decoded_bytes is None if compute_decoded=False.
        """
        input_shape = grouped.shape

        if len(input_shape) == 2:
            # (batch, embedding_size) -> (batch, 4), (batch, 4, 256)
            batch = input_shape[0]
            embeddings = grouped.view(batch, self.max_bytes, self.byte_dim)
            logits = embeddings @ self.embedding.weight.T
            decoded = logits.argmax(dim=-1) if compute_decoded else None
            return decoded, logits

        elif len(input_shape) == 3:
            # (batch, seq, embedding_size) -> (batch, seq, 4), (batch, seq, 4, 256)
            batch, seq, _ = input_shape
            embeddings = grouped.view(batch, seq, self.max_bytes, self.byte_dim)
            logits = embeddings @ self.embedding.weight.T
            decoded = logits.argmax(dim=-1) if compute_decoded else None
            return decoded, logits

        msg = f"Expected 2D or 3D input, got {len(input_shape)}D"
        raise ValueError(msg)

    def forward(self, byte_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode bytes into grouped representation.

        Args:
            byte_indices: Tensor of byte values, shape (batch, 4) or (batch, seq, 4)

        Returns:
            Grouped representation
        """
        return self.encode(byte_indices)
