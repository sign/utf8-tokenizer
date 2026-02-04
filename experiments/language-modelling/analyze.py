import torch
from huggingface_hub import load_state_dict_from_file
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from utf8_tokenizer.byte_embeddings import join_embedding_layers, patch_embedding_layers

MODEL_CHECKPOINT = "./output-tiny-lm-fineweb"


# Load model
config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForCausalLM.from_config(config)
patch_embedding_layers(model)

state_dict = load_state_dict_from_file(f"{MODEL_CHECKPOINT}/model.safetensors")
model.load_state_dict(state_dict)

# Inspect bit projection weights
embeddings = model.get_input_embeddings()
print(embeddings.bit_proj_w.data)

# Save weights to file
torch.save(embeddings.bit_proj_w.data, f"{MODEL_CHECKPOINT}/bit_projection_weights.pt")

# Join embedding layers back
join_embedding_layers(model)
save_file(model.state_dict(), f"{MODEL_CHECKPOINT}/model.safetensors")

model = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT)
