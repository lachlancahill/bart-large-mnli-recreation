from transformers import BartForSequenceClassification, AutoTokenizer, AutoConfig

from datasets import load_dataset
import torch

from sklearn.metrics import accuracy_score
import pandas as pd

import matplotlib.pyplot as plt

# Load Facebook's pre-trained BART-large-MNLI model
config = AutoConfig.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
model_pretrained = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

# Load your custom-trained model
model_custom = BartForSequenceClassification.from_pretrained('./runs/2024-06-13--11-52-22/checkpoints/checkpoint_2', config=config)

# Load the dataset
dataset = load_dataset('glue', 'mnli')
train_dataset = dataset['train']
validation_matched = dataset['validation_matched']
validation_mismatched = dataset['validation_mismatched']


def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation='only_first', padding="max_length",
                        max_length=1024)


train_dataset = train_dataset.map(tokenize_function, batched=True)
validation_matched = validation_matched.map(tokenize_function, batched=True)
validation_mismatched = validation_mismatched.map(tokenize_function, batched=True)

# Assuming you're using PyTorch and a GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_model(model, dataset):
    model.eval()
    predictions = []
    references = []
    for batch in dataset:
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions.extend(logits.argmax(dim=-1).tolist())
        references.extend(batch['label'].tolist())
    accuracy = accuracy_score(references, predictions)
    return accuracy


model_pretrained.to(device)
model_custom.to(device)

# Evaluate on different splits
accuracy_pretrained_train = evaluate_model(model_pretrained, train_dataset)
accuracy_custom_train = evaluate_model(model_custom, train_dataset)
accuracy_pretrained_val_matched = evaluate_model(model_pretrained, validation_matched)
accuracy_custom_val_matched = evaluate_model(model_custom, validation_matched)
accuracy_pretrained_val_mismatched = evaluate_model(model_pretrained, validation_mismatched)
accuracy_custom_val_mismatched = evaluate_model(model_custom, validation_mismatched)

# Assuming you have the accuracy values stored as mentioned in the previous steps
data = {
    'Model': ['BART-large-MNLI', 'Custom BART'],
    'Train Accuracy': [accuracy_pretrained_train, accuracy_custom_train],
    'Validation Matched Accuracy': [accuracy_pretrained_val_matched, accuracy_custom_val_matched],
    'Validation Mismatched Accuracy': [accuracy_pretrained_val_mismatched, accuracy_custom_val_mismatched]
}

results_df = pd.DataFrame(data)
print(results_df.to_string(), "\n")

# Set the style
plt.style.use('ggplot')

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # the width of the bars
ind = [0, 1, 2]  # the x locations for the groups

p1 = ax.bar(ind, results_df['Train Accuracy'], width, bottom=0)
p2 = ax.bar([p + width for p in ind], results_df['Validation Matched Accuracy'], width, bottom=0)
p3 = ax.bar([p + 2 * width for p in ind], results_df['Validation Mismatched Accuracy'], width, bottom=0)

ax.set_title('Model Performance Comparison')
ax.set_xticks([p + width for p in ind])
ax.set_xticklabels(('Train', 'Validation Matched', 'Validation Mismatched'))

ax.legend((p1[0], p2[0], p3[0]), ('BART-large-MNLI', 'Custom BART'))
ax.autoscale_view()

plt.show()
