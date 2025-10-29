# train_simplifier.py
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

print("🔹 Loading pre-trained simplification model (Pegasus paraphraser)...")
model_name = "tuner007/pegasus_paraphrase"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Test simplification
sentence = "Photosynthesis is the process by which green plants utilize sunlight to produce food."
batch = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
translated = model.generate(**batch, max_length=80, num_beams=5, temperature=1.2)
simplified_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)

print("\n✅ Simplified sentence:")
print(simplified_sentence)

# Save model locally
model.save_pretrained("./simplifier_model")
tokenizer.save_pretrained("./simplifier_model")
print("\n💾 Model saved to ./simplifier_model/")
