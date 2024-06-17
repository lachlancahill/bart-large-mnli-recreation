
from transformers import AutoTokenizer, AutoModelForSequenceClassification

hf_repo = 'google/t5-v1_1-large'

tokenizer = AutoTokenizer.from_pretrained(hf_repo)
model = AutoModelForSequenceClassification.from_pretrained(hf_repo)

print(f"{type(tokenizer)=}")
print(f"{type(model)=}")

