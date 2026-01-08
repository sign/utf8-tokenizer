import struct
import sys
import warnings
from collections import namedtuple
from functools import cache
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, TextInput

from utf8_tokenizer.control import ControlTokens
from utf8_tokenizer.utils import pad_bytearrays_to_tensor


@cache
def is_embedding_dtype_supported(dtype: torch.dtype) -> bool:
    try:
        embedding = torch.nn.Embedding(2, 1)
        test_input = torch.tensor([0], dtype=dtype)
        embedding(test_input)
    except (RuntimeError, TypeError):
        return False
    else:
        return True


TokenizerResult = namedtuple("TokenizerResult", ["input_ids", "attention_mask"])

PAD_TOKEN = ControlTokens.Null
PAD_TOKEN_ID = ord(PAD_TOKEN)

BOS_TOKEN = ControlTokens.StartOfText
BOS_TOKEN_ID = ord(BOS_TOKEN)

EOS_TOKEN = ControlTokens.EndOfText
EOS_TOKEN_ID = ord(EOS_TOKEN)

STRUCT_FORMAT: dict[torch.dtype, str] = {
    torch.uint8: "B",
    torch.uint16: "H",
    torch.uint32: "I",
}


class UTFTokenizer(PreTrainedTokenizer):
    """
    Base UTF Tokenizer implementation supporting UTF-8, UTF-16, and UTF-32 encodings.
    Extends PreTrainedTokenizer for Hugging Face ecosystem support.

    Subclasses should set `encoding` and `dtype` class attributes.

    Exposes a `.torch` method for optimized tensor creation.
    """

    __slots__ = ('_struct_format', '_bytes_per_unit', '_eos_encoded')

    encoding: str = "utf-8"
    dtype: torch.dtype = torch.uint8

    def __init__(self, **kwargs):
        self._struct_format = STRUCT_FORMAT[self.dtype]
        self._bytes_per_unit = struct.calcsize(self._struct_format)
        self._eos_encoded = EOS_TOKEN.encode(self.encoding)

        kwargs["pad_token"] = PAD_TOKEN
        kwargs["pad_token_id"] = PAD_TOKEN_ID
        kwargs["bos_token"] = BOS_TOKEN
        kwargs["bos_token_id"] = BOS_TOKEN_ID
        kwargs["eos_token"] = EOS_TOKEN
        kwargs["eos_token_id"] = EOS_TOKEN_ID
        super().__init__(**kwargs)

        # Chat template for instruction-following models
        with open(Path(__file__).parent / "chat_template.jinja") as f:
            self.chat_template = ""
            for line in f:
                self.chat_template += line.strip()

    @property
    def vocab_size(self) -> int:
        # Always 256 because downstream models should use byte-level embedding
        return 256

    def add_tokens(self, *args, **kwargs):
        raise NotImplementedError("UTFTokenizer does not support adding tokens")

    def _tokenize_ids(self, text: str, errors="strict") -> list[int]:
        bytes_per_unit, struct_format = self._bytes_per_unit, self._struct_format
        encoded = text.encode(self.encoding, errors=errors)
        count = len(encoded) // bytes_per_unit
        return list(struct.unpack(f"{count}{struct_format}", encoded))

    def get_vocab(self):
        return {chr(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: TextInput, errors="strict", **kwargs):
        return [chr(c) for c in self._tokenize_ids(text, errors=errors)]

    def tokenize(self, text: TextInput, errors="strict", **kwargs):
        return self._tokenize(text, errors=errors, **kwargs)

    def _encode_plus(self, text: TextInput, **kwargs):
        return self.prepare_for_model(self._tokenize_ids(text), **kwargs)

    def _convert_token_to_id(self, token: str):
        return ord(token)

    def _convert_id_to_token(self, index: int):
        return chr(index)

    def convert_tokens_to_string(self, tokens: list[str]):
        ids = [ord(t) for t in tokens]
        _bytes = struct.pack(f"{len(ids)}{self._struct_format}", *ids)
        return _bytes.decode(self.encoding, errors="ignore")

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int] | bytearray, token_ids_1: list[int] | None = None
    ) -> list[int] | bytearray:
        assert token_ids_1 is None, "UTFTokenizer only supports single sequence"

        token_ids_0.append(EOS_TOKEN_ID)
        token_ids_0.insert(0, BOS_TOKEN_ID)
        return token_ids_0

    def _encode(self, texts: list[TextInput], add_special_tokens: bool) -> list[bytes]:
        encoding = self.encoding
        if add_special_tokens:
            texts = [BOS_TOKEN + text + EOS_TOKEN for text in texts]
        return [text.encode(encoding) for text in texts]

    def _encode_and_truncate(
        self, texts: list[TextInput], max_length: int, add_special_tokens: bool
    ) -> list[bytes]:
        encoding, bytes_per_unit, eos_encoded = self.encoding, self._bytes_per_unit, self._eos_encoded
        # Convert from token count to byte count for slicing
        max_length *= bytes_per_unit
        if add_special_tokens:
            # Include BOS token in the slice, then append EOS
            max_length += bytes_per_unit
            return [(BOS_TOKEN + text).encode(encoding)[:max_length] + eos_encoded for text in texts]
        return [text.encode(encoding)[:max_length] for text in texts]

    def _original_call(self, *args, **kwargs) -> BatchEncoding:
        return super().__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        return_tensors = kwargs.pop("return_tensors", "pt")
        if return_tensors != "pt":
            return self._original_call(*args, return_tensors=return_tensors, **kwargs)
        result = self.torch(*args, **kwargs)
        return result._asdict()

    def torch(
        self,
        texts: list[TextInput],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        device: torch.device | None = None,
    ) -> TokenizerResult:
        if truncation:
            if max_length is None:
                warnings.warn(
                    "Asking to truncate to max_length but no maximum length is provided and the model has "
                    "no predefined maximum length. Default to no truncation.",
                    stacklevel=2,
                )
                truncation = False
            else:
                max_length = max_length - 2 if add_special_tokens else max_length
                if max_length < 0:
                    warnings.warn(
                        "We need to remove more tokens than exist. Default to no truncation.",
                        stacklevel=2)
                    truncation = False

        if truncation:
            input_bytes = self._encode_and_truncate(texts, max_length, add_special_tokens)
        else:
            input_bytes = self._encode(texts, add_special_tokens)

        dtype = self.dtype
        if padding:
            input_ids = pad_bytearrays_to_tensor(input_bytes, dtype, PAD_TOKEN_ID)
            attention_mask = input_ids.ne(PAD_TOKEN_ID)
        else:
            # Slow path - no padding means we need to return a list of tensors
            # bytearray() needed because bytes are immutable -> read-only tensor warning
            input_ids = [torch.frombuffer(bytearray(b), dtype=dtype) for b in input_bytes]
            attention_mask = [torch.ones(len(ids), dtype=torch.bool) for ids in input_ids]

        if not is_embedding_dtype_supported(dtype):
            if isinstance(input_ids, list):
                input_ids = [ids.long() for ids in input_ids]
            else:
                input_ids = input_ids.long()

        if device is not None:
            if isinstance(input_ids, list):
                input_ids = [ids.to(device, non_blocking=True) for ids in input_ids]
                attention_mask = [mask.to(device, non_blocking=True) for mask in attention_mask]
            else:
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)

        return TokenizerResult(input_ids=input_ids, attention_mask=attention_mask)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        return ()

    def to_dict(self):
        return {}


class UTF8Tokenizer(UTFTokenizer):
    encoding = "utf-8"
    dtype = torch.uint8


# System-dependent, but token IDs remain the same across systems (encoding and decoding use matching byte order)
ENDIAN_SUFFIX = "le" if sys.byteorder == "little" else "be"


class UTF16Tokenizer(UTFTokenizer):
    encoding = f"utf-16-{ENDIAN_SUFFIX}"
    dtype = torch.uint16


class UTF32Tokenizer(UTFTokenizer):
    encoding = f"utf-32-{ENDIAN_SUFFIX}"
    dtype = torch.uint32


AutoTokenizer.register(UTF8Tokenizer, slow_tokenizer_class=UTF8Tokenizer)
AutoTokenizer.register(UTF16Tokenizer, slow_tokenizer_class=UTF16Tokenizer)
AutoTokenizer.register(UTF32Tokenizer, slow_tokenizer_class=UTF32Tokenizer)
