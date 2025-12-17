# Back to Bytes: Revisiting Tokenization Through `UTF-8`

![Python](https://img.shields.io/badge/python-3.10-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2510.16987-b31b1b.svg)](https://arxiv.org/abs/2510.16987)

Full writeup can be found in our paper.

This module includes a **real** byte level tokenizer for text, which encodes text into a sequence of bytes (0-255).
Unlike `ByT5Tokenizer` for example, `UTF8Tokenizer` is implemented from scratch, and is much more efficient.

Other "Byte Level" tokenizers usually include various additional "special tokens" (e.g., `<pad>`, `<unk>`, etc.),
making the encoding and decoding logic more complex, and the token ids larger than 255.

Instead, we rely on C0 Control characters (0-31) as special tokens, which are not used in normal text.

## Usage

```shell
pip install utf8-tokenizer
```

Tokenization:

```python
from utf8_tokenizer.tokenizer import UTF8Tokenizer

tokenizer = UTF8Tokenizer()

texts = ["word", "or multiple"]
print(tokenizer(texts))
```

Chat Template:

```py
from utf8_tokenizer.tokenizer import UTF8Tokenizer
from utf8_tokenizer.control import visualize_control_tokens

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hey, what's 1+1?"},
    {"role": "assistant", "content": "1+1 is 2."},
]

tokenizer = UTF8Tokenizer()
text = tokenizer.apply_chat_template(messages, tokenize=False)

# Visualize the text with special tokens
print(visualize_control_tokens(text))
```

Bit-biased byte embeddings:

```py
from transformers import AutoModelForCausalLM

# Load example model
model = AutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm")
model.resize_token_embeddings(256)

from utf8_tokenizer.embeddings import patch_embedding_layers, join_embedding_layers

patch_embedding_layers(model) # Apply bit-bias for training

#
# Train your model...
#

join_embedding_layers(model) # Fold to a single embedding layer for inference
```

UTF-8 Validation during Generation:

```py
from transformers import AutoModelForCausalLM
from utf8_tokenizer import UTF8Tokenizer, UTF8ValidationLogitsProcessor

# Load your byte-level model
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = UTF8Tokenizer()

# Create the UTF-8 validation processor
utf8_processor = UTF8ValidationLogitsProcessor()

# Generate text with UTF-8 validation
input_text = "Hello"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(
    input_ids,
    logits_processor=[utf8_processor],  # Ensures valid UTF-8 sequences
    max_new_tokens=100
)

# Decode the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

The `UTF8ValidationLogitsProcessor` prevents byte-level tokenizers from generating malformed UTF-8 sequences by masking invalid byte continuations during generation. This addresses the issue discussed in [Firestone et al. 2024](https://openreview.net/pdf?id=8ExXncFpf6) where byte-level tokenizers can generate ill-formed UTF-8.

## Benchmark

### Tokenization Speed

```shell
python experiments/benchmark.py
```

On MacBook Pro, with Apple M4 Pro chip, just converting texts of 75 words in different languages to bytes,
without wrapping them in tensors, creating attention masks, or padding, runs at 109.9k/sec.

Calling the ByT5 tokenizer runs at 0.4k/sec.
When we call our new tokenizer, through the `__call__` path, we get 0.5k/sec, which is a bit faster.

Our optimized version with zero-copy runs at 66k/sec, where the loss of performance compared to the raw ints is
in padding the input ids into a properly padded tensor. **This is a 164x speedup over the original tokenizer.**

### Bit-Biased Byte Embedding

We [train a small language model](experiments/language-modelling/README.md) with and without bit-bias.

Our results reveal that bit-bias improves both loss and accuracy, while increasing training time by about 1%.
We hope that our bit-level embeddings module can be further optimized, to minimize the training overhead.

## Cite

If you use this code in your research, please consider citing the work:

```bibtex
@misc{moryossef2025utf8,
  title         = {Back to Bytes: Revisiting Tokenization Through {UTF-8}},
  author        = {Amit Moryossef and Clara Meister and Pavel Stepachev and Desmond Elliott},
  howpublished  = {\url{https://github.com/sign/utf8-tokenizer}},
  eprint        = {2510.16987},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2510.16987}, 
  year          = {2025}
}
```