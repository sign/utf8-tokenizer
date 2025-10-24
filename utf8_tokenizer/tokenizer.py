import warnings
from collections import namedtuple
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, TextInput

from utf8_tokenizer.control import ControlTokens


def tokenize_ids(text: str, errors="strict"):
    return list(text.encode("utf-8", errors=errors))


TokenizerResult = namedtuple("TokenizerResult", ["input_ids", "attention_mask"])

PAD_TOKEN = ControlTokens.Null
PAD_TOKEN_ID = ord(PAD_TOKEN)

BOS_TOKEN = ControlTokens.StartOfText
BOS_TOKEN_ID = ord(BOS_TOKEN)

EOS_TOKEN = ControlTokens.EndOfText
EOS_TOKEN_ID = ord(EOS_TOKEN)


class UTF8Tokenizer(PreTrainedTokenizer):
    """
    Custom UTF8 Byte Level Tokenizer implementation,
    extending PreTrainedTokenizer for basic Hugging Face ecosystem support.

    This tokenizer only supports exactly 256 tokens, with no support for "special tokens".
    See README.md to learn more how this works with control tokens instead.

    Additionally, exposes a `.torch` method, which fuses and skips unnecessary ops,
    to achieve a ~8x speedup over `__call__` for training purposes.
    """

    def __init__(self, **kwargs):
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
        return 2 ** 8

    def add_tokens(self, *args, **kwargs):
        raise NotImplementedError("UTF8Tokenizer does not support adding tokens")

    def get_vocab(self):
        return {chr(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: TextInput, errors="strict", **kwargs):
        return [chr(c) for c in tokenize_ids(text, errors=errors)]

    def tokenize(self, text: TextInput, errors="strict", **kwargs):
        return self._tokenize(text, errors=errors, **kwargs)

    def _encode_plus(self, text: TextInput, **kwargs):
        return self.prepare_for_model(tokenize_ids(text), **kwargs)

    def _convert_token_to_id(self, token: str):
        return ord(token)

    def _convert_id_to_token(self, index: int):
        return chr(index)

    def convert_tokens_to_string(self, tokens: list[str]):
        """Converts a sequence of tokens (string) in a single string."""
        _bytes = bytes(ord(token) for token in tokens)
        return _bytes.decode("utf-8", errors="ignore")

    def build_inputs_with_special_tokens(
            self, token_ids_0: list[int] | bytearray, token_ids_1: list[int] | None = None
    ) -> list[int] | bytearray:
        assert token_ids_1 is None, "UTF8Tokenizer only supports single sequence"

        # Experimentally, the fastest way to add BOS/EOS
        token_ids_0.append(EOS_TOKEN_ID)  # EOS
        token_ids_0.insert(0, BOS_TOKEN_ID)  # BOS
        return token_ids_0

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
        input_bytes = [bytearray(text, "utf-8") for text in texts]

        if truncation:
            if max_length is None:
                warnings.warn(
                    "Asking to truncate to max_length but no maximum length is provided and the model has "
                    "no predefined maximum length. Default to no truncation.",
                    stacklevel=2,
                )
            else:
                corrected_max_length = max_length - 2 if add_special_tokens else max_length
                if corrected_max_length < 0:
                    warnings.warn("We need to remove more tokens than exist. Default to no truncation.", stacklevel=2)
                else:
                    input_bytes = [text_bytes[:corrected_max_length] for text_bytes in input_bytes]

        if add_special_tokens:
            # Faster to manipulate strings than lists of ints
            input_bytes = [self.build_inputs_with_special_tokens(ids) for ids in input_bytes]

        # torch.frombuffer is faster than torch.tensor for bytearrays since no copy is made
        input_ids = [torch.frombuffer(ids, dtype=torch.uint8) for ids in input_bytes]

        if padding:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
            attention_mask = input_ids.ne(0)
        else:
            # Slow path - no padding means we need to return a list of tensors
            attention_mask = [torch.ones(len(ids), dtype=torch.bool) for ids in input_ids]

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


AutoTokenizer.register(UTF8Tokenizer, slow_tokenizer_class=UTF8Tokenizer)
