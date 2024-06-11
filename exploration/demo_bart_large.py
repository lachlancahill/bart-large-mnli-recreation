import torch
from transformers import BartTokenizer, BartModel
from config import device

# Load the tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartModel.from_pretrained('facebook/bart-large').to(device)

# Tokenize the input text
inputs = tokenizer("Hello, my dog is so", return_tensors="pt").to(device)

# Get the model outputs
outputs = model(**inputs)

# Get the logits for the last token in the input sequence
logits = outputs.last_hidden_state[:, -1, :]

# Apply softmax to get probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)

# Get the token with the highest probability
predicted_token_id = torch.argmax(probs, dim=-1)

# Decode the predicted token
predicted_token = tokenizer.decode(predicted_token_id)

print(predicted_token)