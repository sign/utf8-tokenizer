# CausalLMWrapper Performance

Performance comparison between `CausalLMWrapper` and the base `AutoModelForCausalLM`.

## Test Setup

- **Model**: `sign/utf8-lm-tiny` (hidden_size=256)
- **Device**: CPU (Apple Silicon)
- **Batch size**: 1
- **Methodology**: 10 trials × 50 runs each, reporting mean ± std

## How It Works

`CausalLMWrapper` groups UTF-8 bytes into 4-byte chunks, reducing sequence length:

| Text Type      | Bytes/Char | Compression |
|----------------|------------|-------------|
| ASCII          | 1          | 1.0x        |
| Extended Latin | 2          | 2.0x        |
| CJK            | 3          | 3.0x        |
| Emoji          | 4          | 4.0x        |

## Forward Pass

| Text        | Bytes | Groups | Compression | Base   | Wrapper | Speedup   |
|-------------|-------|--------|-------------|--------|---------|-----------|
| English 128 | 130   | 130    | 1.0x        | 5.5ms  | 5.8ms   | 0.96x     |
| English 256 | 258   | 258    | 1.0x        | 7.7ms  | 8.1ms   | 0.95x     |
| English 512 | 514   | 514    | 1.0x        | 11.1ms | 11.9ms  | 0.93x     |
| Chinese 128 | 386   | 130    | 3.0x        | 9.7ms  | 6.7ms   | **1.45x** |
| Chinese 256 | 770   | 258    | 3.0x        | 15.9ms | 9.4ms   | **1.68x** |
| Chinese 512 | 1538  | 514    | 3.0x        | 32.2ms | 14.1ms  | **2.29x** |
| Mixed 64+64 | 258   | 130    | 2.0x        | 10.0ms | 7.8ms   | **1.29x** |
| Emoji 64    | 258   | 66     | 3.9x        | 10.0ms | 3.9ms   | **2.55x** |

For multi-byte UTF-8 text, the wrapper is **1.3-2.6x faster**.

## Decode Step (Autoregressive)

| Metric         | Base Model | Wrapper   | Comparison      |
|----------------|------------|-----------|-----------------|
| Time per step  | 2.12ms     | 2.03ms    | ~same           |
| Bytes per step | 1          | 4         | 4x more         |
| Throughput     | 472 B/s    | 1,966 B/s | **4.2x faster** |

The wrapper has no overhead per step while producing 4x more bytes.

## Generation

| Target    | Base             | Wrapper         | Speedup  |
|-----------|------------------|-----------------|----------|
| 32 bytes  | 32 steps, 26ms   | 8 groups, 7.5ms | **3.5x** |
| 64 bytes  | 64 steps, 53ms   | 16 groups, 14ms | **3.9x** |
| 128 bytes | 128 steps, 109ms | 32 groups, 27ms | **4.1x** |
| 256 bytes | 256 steps, 253ms | 64 groups, 53ms | **4.8x** |

## Summary

| Operation  | ASCII               | Multi-byte UTF-8    |
|------------|---------------------|---------------------|
| Forward    | 0.93-0.96x          | **1.3-2.6x faster** |
| Generation | **3.5-4.8x faster** | **3.5-4.8x faster** |

The wrapper is recommended for all text generation tasks.
