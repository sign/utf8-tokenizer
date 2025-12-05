import numpy as np
import torch


def pad_bytearrays_to_tensor_loop(bytearrays: list[bytearray], padding_value: int = 0) -> torch.Tensor:
    """
    Pad a list of bytearrays into a single tensor using a simple loop.

    This is the non-vectorized reference implementation.
    """
    max_len = max(len(b) for b in bytearrays)
    output = np.full((len(bytearrays), max_len), padding_value, dtype=np.uint8)

    for i, b in enumerate(bytearrays):
        output[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)

    return torch.from_numpy(output)


def pad_bytearrays_to_tensor(bytearrays: list[bytearray], padding_value: int = 0) -> torch.Tensor:
    """
    Pad a list of bytearrays into a single tensor using vectorized numpy ops.

    This is the optimized implementation that avoids Python loops for index creation.
    Falls back to loop implementation for edge cases with empty bytearrays.
    """
    lengths = np.array([len(b) for b in bytearrays])
    # Handle edge case: empty bytearrays - fall back to loop
    if (lengths == 0).any():
        return pad_bytearrays_to_tensor_loop(bytearrays, padding_value)

    max_len = lengths.max()
    batch_size = len(bytearrays)

    # Pre-allocate output
    output = np.full((batch_size, max_len), padding_value, dtype=np.uint8)

    # Concatenate all bytes into single buffer, single frombuffer call
    all_bytes = b''.join(bytearrays)
    all_values = np.frombuffer(all_bytes, dtype=np.uint8)

    # Fully vectorized index creation
    row_indices = np.repeat(np.arange(batch_size), lengths)
    # col_indices: [0,1,2,0,1,0,1,2,3,...] for lengths [3,2,4,...]
    offsets = np.zeros(len(all_values), dtype=np.int32)
    offsets[lengths.cumsum()[:-1]] = lengths[:-1]
    col_indices = np.arange(len(all_values)) - offsets.cumsum()

    # Single vectorized assignment
    output[row_indices, col_indices] = all_values

    return torch.from_numpy(output)
