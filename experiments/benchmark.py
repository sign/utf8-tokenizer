import torch
from tqdm import tqdm

from utf8_tokenizer.byt5_comparison import ByT5ComparableTokenizer
from utf8_tokenizer.tokenizer import UTF8Tokenizer, tokenize_ids

if __name__ == "__main__":
    tokenizer = UTF8Tokenizer()
    texts = [
        "Hello", "World", "the", "a", "is", "of", "and", "to", "in", "that",
        "שלום", "עולם", "את", "של", "על", "עם", "לא", "הוא", "היא", "זה",
        "I", "you", "we", "they", "it", "be", "have", "do", "say", "get",
        "make", "go", "know", "take", "see", "come", "think", "look", "want", "give",
        "use", "find", "tell", "ask", "work", "seem", "feel", "try", "leave", "call",
        "<en>", "<he>", "\x0E", "\x0F",  # Special tokens
        ".", ",", "!", "?", ":", ";", "-", "(", ")", '"',
        "hello!", "world.", "test,", "foo-bar", "(test)", '"quoted"',
        "אבגדהוזחטיכלמנסעפצקרשת",  # Hebrew alphabet
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # English alphabet
        "0123456789",  # Numbers
        " ",  # Space
        "emoji🤗",
    ]

    num = 1000

    for _ in tqdm(range(num), desc="just tokenizing to ints"):
        for text in texts:
            tokenize_ids(text)

    for _ in tqdm(range(num), desc="Calling the new tokenizer"):
        tokenizer(texts, add_special_tokens=True, padding=True, return_tensors="pt")

    for _ in tqdm(range(num), desc="Calling the new tokenizer.torch"):
        tokenizer.torch(texts, add_special_tokens=True, padding=True)

    for _ in tqdm(range(num), desc="Calling the new tokenizer._original_call"):
        tokenizer._original_call(texts, add_special_tokens=True, padding=True, return_tensors="pt")

    tokenizer = ByT5ComparableTokenizer()
    for _ in tqdm(range(num), "Calling the old tokenizer"):
        tokenizer(texts, add_special_tokens=True, padding=True, return_tensors="pt")
