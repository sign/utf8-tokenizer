# Back to Bytes: Revisiting Tokenization Through `UTF-8`

Full writeup can be found in the paper.

This module includes a **real** byte level tokenizer for text, which encodes text into a sequence of bytes (0-255).
Unlike `ByT5Tokenizer` for example, `UTF8Tokenizer` is implemented from scratch, and is much more efficient.

Other "Byte Level" tokenizers usually include various additional "special tokens" (e.g., `<pad>`, `<unk>`, etc.),
making the encoding and decoding logic more complex, and the token ids larger than 255.

Instead, we rely on C0 Control characters (0-31) as special tokens, which are not used in normal text.

## Usage

Tokenization:

```python
from utf8_tokenizer.tokenizer import UTF8Tokenizer

tokenizer = UTF8Tokenizer()

texts = ["word", "or multiple"]
print(tokenizer(texts))

# Very fast version
print(tokenizer.torch(texts))
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

## Benchmark

### Tokenization Speed

```shell
python experiments/benchmark.py
```

On MacBook Pro, with Apple M4 Pro chip, just converting texts of 6 words in different languages to bytes, 
without wrapping them in tensors, creating attention masks, or padding, runs at 127.4k/sec.

Calling the ByT5 tokenizer runs at 6.2k/sec.
When we call our new tokenizer, through the `__call__` path, we get 10.5k/sec, which is a bit faster.

Our optimized version with zero-copy runs at 86.7k/sec, where the loss of performance compared to the raw ints is 
in padding the input ids into a properly padded tensor. **This is a 14x speedup over the original tokenizer.**

### Bit-Biased Byte Embedding

We [train a small language model](experiments/language-modelling/README.md) with and without bit-bias.

Our results reveal that bit-bias improves both loss and accuracy, while increasing training time by about 1%.
We hope that our bit-level embeddings module can be further optimized, to minimize the training overhead.

## Cite

If you use this code in your research, please consider citing the work:

```bibtex
@misc{moryossef2025utf8,
  title={Back to Bytes: Revisiting Tokenization Through {UTF-8}},
  author={Moryossef, Amit},
  howpublished={\url{https://github.com/sign/utf8-tokenizer}},
  year={2025}
}
```