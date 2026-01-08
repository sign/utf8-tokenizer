from transformers import AutoModelForCausalLM, LogitsProcessorList
import torch
from utf8_tokenizer.logits_processor import UTF8ValidationLogitsProcessor
from utf8_tokenizer.groups.causal_lm import GroupedCausalLMWrapper

from utf8_tokenizer import UTF8Tokenizer

# model_id = "sign/utf8-lm-tiny"
model_id = "sign/utf8-groups-lm-tiny"

tokenizer = UTF8Tokenizer()
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "My name is Barack"

inputs = tokenizer([prompt], return_tensors="pt",
                   padding=True,
                   add_special_tokens=True)
# We need to remove the EOS token
inputs["input_ids"] = inputs["input_ids"][:, :-1]
inputs["attention_mask"] = inputs["attention_mask"][:, :-1]


with torch.no_grad():
    out = model.generate(
        **inputs,
        logits_processor=LogitsProcessorList([UTF8ValidationLogitsProcessor()]),
        max_new_tokens=256,
    )

print(tokenizer.decode(out[0], skip_special_tokens=False))
