import torch
from tqdm import tqdm

from utf8_tokenizer.byt5_comparison import ByT5ComparableTokenizer
from utf8_tokenizer.tokenizer import UTF8Tokenizer, tokenize_ids


if __name__ == "__main__":
    tokenizer = UTF8Tokenizer()
    texts = ["a", "few", "words", "emoji🤗", "עברית", "中文"]

    num = 10000

    for _ in tqdm(range(num), desc="just tokenizing to ints"):
        for text in texts:
            torch.tensor(tokenize_ids(text), dtype=torch.long)

    for _ in tqdm(range(num), desc="Calling the new tokenizer"):
        tokenizer(texts, add_special_tokens=True, padding=True, return_tensors="pt")

    for _ in tqdm(range(num), desc="Calling the new tokenizer.torch"):
        tokenizer.torch(texts, add_special_tokens=True, padding=True)

    for _ in tqdm(range(num), desc="Calling the new tokenizer._original_call"):
        tokenizer._original_call(texts, add_special_tokens=True, padding=True, return_tensors="pt")

    tokenizer = ByT5ComparableTokenizer()
    for _ in tqdm(range(num), "Calling the old tokenizer"):
        tokenizer(texts, add_special_tokens=True, padding=True, return_tensors="pt")
