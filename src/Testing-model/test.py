from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./trained_model")
model = AutoModelForCausalLM.from_pretrained("./trained_model")

prompt = "What is my name?"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=50
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))