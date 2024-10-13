from transformers import AutoTokenizer, MistralForCausalLM
from model_config import model_paths
model = MistralForCausalLM.from_pretrained(model_paths["mistral-7b"])
tokenizer = AutoTokenizer.from_pretrained(model_paths["mistral-7b"])

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."