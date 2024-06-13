import tqdm
from transformers import BartForSequenceClassification, AutoTokenizer, AutoConfig, BartTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate


device = torch.device('cuda:0')

# Load Facebook's pre-trained BART-large-MNLI model
tokenizer:BartTokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
model_pretrained = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

# Load your custom-trained model
config = AutoConfig.from_pretrained('facebook/bart-large-mnli')
model_custom = BartForSequenceClassification.from_pretrained('./runs/2024-06-13--11-52-22/checkpoints/checkpoint_2', config=config)

# Load the dataset
dataset = load_dataset('glue', 'mnli')
train_dataset = dataset['train']
validation_matched = dataset['validation_matched']
validation_mismatched = dataset['validation_mismatched']

def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation='only_first', padding="max_length", max_length=1024)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "premise", "hypothesis"]).rename_column("label", "labels")
validation_matched = validation_matched.map(tokenize_function, batched=True, remove_columns=["idx", "premise", "hypothesis"]).rename_column("label", "labels")
validation_mismatched = validation_mismatched.map(tokenize_function, batched=True, remove_columns=["idx", "premise", "hypothesis"]).rename_column("label", "labels")

train_dataset.set_format('torch')
validation_matched.set_format('torch')
validation_mismatched.set_format('torch')

def collate_fn(batch):
    default_collated_batch = default_collate(batch)
    correct_device_batch = {k:v.to(device) for k,v in default_collated_batch.items()}
    return correct_device_batch
    # return default_collated_batch
    # return batch
    # return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

eval_batch_size = 64
# collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))


# train_loader = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=True, collate_fn=collate_fn)
validation_matched_loader = DataLoader(validation_matched, batch_size=eval_batch_size, collate_fn=collate_fn)
validation_mismatched_loader = DataLoader(validation_mismatched, batch_size=eval_batch_size, collate_fn=collate_fn)


def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    references = []

    for batch in tqdm.tqdm(dataloader, desc='Evaluating'):
        # inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions.extend(logits.argmax(dim=-1).tolist())
        references.extend(batch['labels'].tolist())
    accuracy = accuracy_score(references, predictions)
    return accuracy


model_pretrained.to(device)
model_custom.to(device)

# Evaluate on different splits
# accuracy_pretrained_train = evaluate_model(model_pretrained, train_loader)
# accuracy_custom_train = evaluate_model(model_custom, train_loader)
accuracy_pretrained_val_matched = evaluate_model(model_pretrained, validation_matched_loader)
accuracy_custom_val_matched = evaluate_model(model_custom, validation_matched_loader)
accuracy_pretrained_val_mismatched = evaluate_model(model_pretrained, validation_mismatched_loader)
accuracy_custom_val_mismatched = evaluate_model(model_custom, validation_mismatched_loader)

# Assuming you have the accuracy values stored as mentioned in the previous steps
data = {
    'Model': ['BART-large-MNLI', 'Custom BART'],
    # 'Train Accuracy': [accuracy_pretrained_train, accuracy_custom_train],
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