import numpy as np
import torch

try:
    from numba import jit

    @jit(nopython=True)
    def _fill_padded(output: np.ndarray, all_values: np.ndarray, lengths: np.ndarray, pad_value: int) -> None:
        """Numba JIT-compiled loop to fill values and padding in single pass."""
        batch_size, max_len = output.shape
        offset = 0
        for i in range(batch_size):
            length = lengths[i]
            # Fill data
            for j in range(length):
                output[i, j] = all_values[offset + j]
            # Fill padding
            for j in range(length, max_len):
                output[i, j] = pad_value
            offset += length

except ImportError:
    def _fill_padded(output: np.ndarray, all_values: np.ndarray, lengths: np.ndarray, pad_value: int) -> None:
        """Fill padded output array using vectorized numpy ops."""
        output.fill(pad_value)
        if len(all_values) == 0:
            return
        batch_size = len(lengths)
        row_indices = np.repeat(np.arange(batch_size), lengths)
        cumsum = lengths.cumsum()
        positions = np.arange(len(all_values))
        groups = np.searchsorted(cumsum, positions, side='right')
        prev_cumsum = np.empty(batch_size, dtype=np.uint32)
        prev_cumsum[0] = 0
        prev_cumsum[1:] = cumsum[:-1]
        col_indices = positions - prev_cumsum[groups]
        output[row_indices, col_indices] = all_values


def pad_bytearrays_to_tensor_loop(bytearrays: list[bytearray], padding_value: int = 0) -> torch.Tensor:
    """
    Pad a list of bytearrays into a single tensor using a simple loop.

    This is the reference implementation for testing.
    """
    max_len = max(len(b) for b in bytearrays)
    output = np.full((len(bytearrays), max_len), padding_value, dtype=np.uint8)

    for i, b in enumerate(bytearrays):
        output[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)

    return torch.from_numpy(output)


def pad_bytearrays_to_tensor(bytearrays: list[bytes], padding_value: int = 0) -> torch.Tensor:
    """
    Pad a list of byte sequences into a single tensor.

    Uses Numba JIT if available, otherwise falls back to vectorized numpy.
    """
    lengths = np.fromiter(map(len, bytearrays), dtype=np.uint32, count=len(bytearrays))
    output = np.empty((len(bytearrays), lengths.max()), dtype=np.uint8)

    all_bytes = b''.join(bytearrays)
    all_values = np.frombuffer(all_bytes, dtype=np.uint8)

    _fill_padded(output, all_values, lengths, padding_value)

    return torch.from_numpy(output)
