"""
Character Embeddings for UTF-16 and UTF-32 tokens.

Provides encode/decode between character tokens and grouped byte embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from utf8_tokenizer.byte_embeddings import PatchedBitEmbeddings


class CharacterEmbedding(nn.Module):
    """
    Character embedding layer that splits multi-byte tokens into byte components.

    Encodes UTF-16 (2 bytes) or UTF-32 (4 bytes) tokens into grouped byte embeddings
    and can decode back to the original token indices.

    Args:
        embedding_size: Total dimension of the output embedding (must be divisible by num_bytes)
        num_bytes: Number of bytes per token (2 for UTF-16, 4 for UTF-32)

    Raises:
        ValueError: If num_bytes is 1 (use a normal embedding layer instead)
        ValueError: If num_bytes not in {2, 4}
        ValueError: If embedding_size is not divisible by num_bytes
    """

    def __init__(self, embedding_size: int, num_bytes: int):
        super().__init__()

        if num_bytes == 1:
            raise ValueError("num_bytes=1 is not supported. Use a normal embedding layer instead.")

        if num_bytes not in (2, 4):
            raise ValueError(f"num_bytes must be 2 or 4, got {num_bytes}")

        if embedding_size % num_bytes != 0:
            raise ValueError(f"embedding_size must be divisible by {num_bytes}, got {embedding_size}")

        self.embedding_size = embedding_size
        self.num_bytes = num_bytes
        self.byte_dim = embedding_size // num_bytes
        self.num_embeddings = 256

        self.embedding = nn.Embedding(self.num_embeddings, self.byte_dim)

        with torch.no_grad():
            # Initialize embedding weights with normalized rows for roundtrip
            self.embedding.weight.copy_(functional.normalize(self.embedding.weight, p=2, dim=1))

        self.embedding = PatchedBitEmbeddings(self.embedding)

        # Pre-compute shift amounts as buffers (0, 8, 16, 24 for the bytes)
        byte_shifts = torch.arange(0, 8 * num_bytes, 8, dtype=torch.long)
        self.register_buffer("_byte_shifts", byte_shifts, persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        """Get the embedding weight matrix."""
        return self.embedding.weight

    @torch.compile()
    def _split_to_bytes(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Split tokens into individual bytes using broadcast bit shifts.

        Args:
            tokens: Token indices with shape (batch,) or (batch, seq)

        Returns:
            Byte indices with shape (batch, num_bytes) or (batch, seq, num_bytes)
        """
        return (tokens.unsqueeze(-1) >> self._byte_shifts) & 0xFF

    @torch.compile()
    def _combine_from_bytes(self, byte_indices: torch.Tensor) -> torch.Tensor:
        """
        Combine bytes back into tokens using broadcast bit shifts.

        Args:
            byte_indices: Byte indices with shape (..., num_bytes)

        Returns:
            Token indices with shape (...)
        """
        return (byte_indices << self._byte_shifts).sum(dim=-1)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode token indices to grouped byte embeddings.

        Args:
            tokens: Token indices (uint16 for UTF-16, uint32 for UTF-32)
                - Shape (batch,) -> returns (batch, embedding_size)
                - Shape (batch, seq) -> returns (batch, seq, embedding_size)

        Returns:
            Grouped embeddings
        """
        input_shape = tokens.shape
        byte_indices = self._split_to_bytes(tokens)
        embeddings = self.embedding(byte_indices)

        if len(input_shape) == 1:
            return embeddings.view(input_shape[0], -1)

        if len(input_shape) == 2:
            batch, seq = input_shape
            return embeddings.view(batch, seq, -1)

        raise ValueError(f"Expected 1D or 2D input, got {len(input_shape)}D")

    def decode(
            self, grouped: torch.Tensor, compute_decoded: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Decode grouped representation back to token indices.

        Args:
            grouped: Grouped representation
                - Shape (batch, embedding_size) -> returns (batch,), (batch, num_bytes, 256)
                - Shape (batch, seq, embedding_size) -> returns (batch, seq), (batch, seq, num_bytes, 256)
            compute_decoded: If True, compute argmax to get decoded tokens.
                If False, return None for decoded (faster for training).

        Returns:
            Tuple of (decoded_tokens, logits). decoded_tokens is None if compute_decoded=False.
        """
        input_shape = grouped.shape

        if len(input_shape) == 2:
            batch = input_shape[0]
            embeddings = grouped.view(batch, self.num_bytes, self.byte_dim)
        elif len(input_shape) == 3:
            batch, seq, _ = input_shape
            embeddings = grouped.view(batch, seq, self.num_bytes, self.byte_dim)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {len(input_shape)}D")

        logits = embeddings @ self.embedding.weight.T

        if self.num_bytes == 4:
            # Apply UTF-32 byte restrictions: top 11 bits must be zero (<= 0x10FFFF)
            logits[..., 3, 1:] = float("-inf")  # 4th byte can only be 0x00
            logits[..., 2, 0x11:] = float("-inf")  # 3rd byte only valid up to 0x10

        if compute_decoded:
            byte_indices = logits.argmax(dim=-1)
            decoded = self._combine_from_bytes(byte_indices)
        else:
            decoded = None

        return decoded, logits

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode tokens into grouped representation.

        Args:
            tokens: Token indices, shape (batch,) or (batch, seq)

        Returns:
            Grouped representation
        """
        return self.encode(tokens)
