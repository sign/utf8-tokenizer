import warnings

from transformers import ByT5Tokenizer


class ByT5ComparableTokenizer(ByT5Tokenizer):
    def __init__(self, *args, **kwargs):
        kwargs["unk_token"] = kwargs.get("unk_token", "<bos>")
        kwargs["bos_token"] = kwargs.get("bos_token", kwargs["unk_token"])
        # Aim for 256 bytes + 3 special tokens + 5 extra ids
        kwargs["extra_ids"] = kwargs.get("extra_ids", 5)
        super().__init__(*args, **kwargs)

    def _add_eos_if_not_present(self, token_ids: list[int]) -> list[int]:
        token_ids = super()._add_eos_if_not_present(token_ids)

        # ByT5Tokenizer does not add BOS token by default, so we add it here
        if len(token_ids) > 0 and token_ids[0] == self.bos_token_id:
            warnings.warn(
                f"This sequence already has {self.bos_token_id}. In future versions this behavior may lead to "
                f"duplicated bos tokens being added.",
                stacklevel=2,
            )
            return token_ids

        return [self.bos_token_id] + token_ids
