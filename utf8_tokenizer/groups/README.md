# UTF-8 Grouping

Converts UTF-8 byte sequences into grouped 4-byte representations, where each group represents one Unicode character.

## What it does

UTF-8 characters use 1-4 bytes. This module groups consecutive bytes belonging to the same character and right-pads each
group to 4 bytes:

```
Input bytes:  [97, 215, 169, 230, 132, 155, 0, 0]
               'a'  '֩' (2 bytes)  '我' (3 bytes)   padding

Output groups: [[0, 0, 0, 97],      # 'a'  - 1 byte, padded left with 3 zeros
                [0, 0, 215, 169],   # '֩'  - 2 bytes, padded left with 2 zeros
                [0, 230, 132, 155]] # '我' - 3 bytes, padded left with 1 zero
```

## Usage

```python
from utf8_tokenizer.groups import group_utf8_bytes

# Works on CPU (PyTorch) or GPU (CUDA kernels, 8x faster)
result = group_utf8_bytes(byte_tensor)  # (batch, seq_len) -> (batch, num_groups, 4)
```

## CUDA Kernels

Two fused CUDA kernels eliminate kernel launch overhead (~8x faster than PyTorch):

| Kernel             | File          | Seq Length | Memory | Use Case       |
|--------------------|---------------|------------|--------|----------------|
| `kernel_shared.cu` | Shared memory | ≤ 1024     | Fast   | Most inputs    |
| `kernel_global.cu` | Global memory | ≤ 64K      | Slower | Long sequences |

Kernel selection is automatic based on sequence length.

## Files

```
groups/
├── __init__.py           # Public API
├── group_utf8_bytes.py   # Main function (dispatches to PyTorch or CUDA)
├── embedding.py          # UTF8GroupedEmbedding nn.Module
├── kernel_shared.cu      # CUDA kernel using shared memory
└── kernel_global.cu      # CUDA kernel using global memory
```
