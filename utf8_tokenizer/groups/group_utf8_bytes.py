"""
UTF-8 byte grouping function.

Converts UTF-8 byte sequences to grouped representation.
Automatically uses CUDA kernels when available for GPU tensors.
"""

from pathlib import Path

import torch

# CUDA kernel state
_kernel_shared = None
_kernel_global = None
_load_attempted_shared = False
_load_attempted_global = False

SHARED_MEM_SEQ_LIMIT = 1024
GLOBAL_MEM_SEQ_LIMIT = 65536


def _load_kernel(name: str, source_file: str):
    """JIT compile a CUDA kernel."""
    if not torch.cuda.is_available():
        return None

    from torch.utils.cpp_extension import load

    current_dir = Path(__file__).parent
    cuda_source = current_dir / source_file

    if not cuda_source.exists():
        return None

    try:
        return load(
            name=name,
            sources=[str(cuda_source)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except (OSError, RuntimeError, ImportError):
        return None


def _get_kernel_shared():
    """Get shared memory kernel (lazy load)."""
    global _kernel_shared, _load_attempted_shared
    if not _load_attempted_shared:
        _load_attempted_shared = True
        _kernel_shared = _load_kernel("group_utf8_shared", "kernel_shared.cu")
    return _kernel_shared


def _get_kernel_global():
    """Get global memory kernel (lazy load)."""
    global _kernel_global, _load_attempted_global
    if not _load_attempted_global:
        _load_attempted_global = True
        _kernel_global = _load_kernel("group_utf8_global", "kernel_global.cu")
    return _kernel_global


def _group_utf8_bytes_torch(byte_indices: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    """
    Pure PyTorch implementation of UTF-8 byte grouping.

    Groups consecutive bytes into UTF-8 groups (leading byte + continuations),
    left-padded with zeros to 4 bytes each.

    Args:
        byte_indices: Tensor of UTF-8 bytes, shape (batch, seq_len)
        padding_value: Value used for padding in input (will be skipped)

    Returns:
        Tensor of shape (batch, max_groups, 4) with left-padded UTF-8 groups
    """
    batch_size, seq_len = byte_indices.shape
    device = byte_indices.device
    dtype = byte_indices.dtype

    # Masks
    non_padding = byte_indices != padding_value
    # Leading bytes: ASCII (0-127) or multi-byte start (192-255), not continuation (128-191)
    is_leading = ((byte_indices < 128) | (byte_indices >= 192)) & non_padding

    # Count groups per batch element
    groups_per_batch = is_leading.sum(dim=1)
    max_groups = groups_per_batch.max()

    if max_groups == 0:
        return torch.zeros(batch_size, 0, 4, dtype=dtype, device=device)

    max_groups = int(max_groups.item())

    # Group index for each byte (0-indexed): cumsum of leading bytes - 1
    group_idx = torch.cumsum(is_leading.int(), dim=1) - 1
    group_idx = group_idx.clamp(min=0)

    # Position within group
    running_count = torch.cumsum(non_padding.int(), dim=1)

    # Get the running_count at each group's start using scatter_reduce min
    large_val = seq_len + 1
    running_at_pos = torch.where(is_leading, running_count, large_val)

    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(group_idx)
    group_start_count = torch.full((batch_size, max_groups), large_val, dtype=torch.int32, device=device)

    # Flatten for scatter
    flat_batch = batch_idx[non_padding]
    flat_group = group_idx[non_padding]
    flat_running = running_at_pos[non_padding].int()

    flat_idx = flat_batch * max_groups + flat_group
    group_start_flat = torch.full((batch_size * max_groups,), large_val, dtype=torch.int32, device=device)
    group_start_flat.scatter_reduce_(0, flat_idx, flat_running, reduce='amin', include_self=True)
    group_start_count = group_start_flat.view(batch_size, max_groups)

    # Position within group
    group_start_at_pos = group_start_count[batch_idx, group_idx]
    pos_in_group = running_count - group_start_at_pos

    # Compute group lengths
    group_lengths = torch.zeros(batch_size, max_groups, dtype=torch.int32, device=device)
    ones = torch.ones_like(flat_batch, dtype=torch.int32)
    group_lengths.view(-1).scatter_add_(0, flat_idx, ones)
    group_lengths = group_lengths.clamp(max=4)

    # Output position: right-aligned
    group_len_at_pos = group_lengths[batch_idx, group_idx]
    out_pos = 4 - group_len_at_pos + pos_in_group

    # Filter valid positions
    valid = non_padding & (pos_in_group < 4) & (out_pos >= 0) & (out_pos < 4)

    # Create output tensor
    output = torch.zeros(batch_size, max_groups, 4, dtype=dtype, device=device)

    # Scatter bytes into output
    flat_out_batch = batch_idx[valid]
    flat_out_group = group_idx[valid]
    flat_out_pos = out_pos[valid]
    flat_out_vals = byte_indices[valid]

    flat_out_idx = flat_out_batch * (max_groups * 4) + flat_out_group * 4 + flat_out_pos
    output.view(-1).scatter_(0, flat_out_idx, flat_out_vals)

    return output


def _group_utf8_bytes_cuda(byte_indices: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    """CUDA kernel implementation (auto-selects shared vs global memory)."""
    seq_len = byte_indices.shape[1]

    # Try shared memory kernel first (fast, seq_len <= 1024)
    if seq_len <= SHARED_MEM_SEQ_LIMIT:
        kernel = _get_kernel_shared()
        if kernel is not None:
            return kernel.group_utf8_bytes_cuda(byte_indices, padding_value)

    # Try global memory kernel (slower, seq_len <= 64K)
    if seq_len <= GLOBAL_MEM_SEQ_LIMIT:
        kernel = _get_kernel_global()
        if kernel is not None:
            return kernel.group_utf8_bytes_cuda(byte_indices, padding_value)

    # Fallback to PyTorch
    return _group_utf8_bytes_torch(byte_indices, padding_value)


def group_utf8_bytes(byte_indices: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    """
    Convert UTF-8 byte sequences to grouped representation.

    Automatically uses fused CUDA kernels for GPU tensors (8x faster),
    falls back to PyTorch for CPU tensors.

    Args:
        byte_indices: Tensor of UTF-8 bytes, shape (batch, seq_len)
        padding_value: Value used for padding in input (default 0)

    Returns:
        Tensor of shape (batch, max_groups, 4) with left-padded UTF-8 groups

    Example:
        >>> x = torch.tensor([[97, 215, 169, 230, 132, 155, 0, 0]])
        >>> group_utf8_bytes(x)
        tensor([[[0, 0, 0, 97], [0, 0, 215, 169], [0, 230, 132, 155]]])
    """
    if byte_indices.is_cuda:
        return _group_utf8_bytes_cuda(byte_indices, padding_value)
    return _group_utf8_bytes_torch(byte_indices, padding_value)


def is_cuda_kernel_available() -> bool:
    """Check if CUDA kernels are available."""
    if not torch.cuda.is_available():
        return False
    return _get_kernel_shared() is not None or _get_kernel_global() is not None
