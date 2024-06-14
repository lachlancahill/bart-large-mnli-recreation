import tqdm
from transformers import BartForSequenceClassification, AutoTokenizer, AutoConfig, BartTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate


eval_training_data = True

device = torch.device('cuda:0')

# Load Facebook's pre-trained BART-large-MNLI model
tokenizer:BartTokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
model_pretrained = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

# Load your custom-trained model
config = AutoConfig.from_pretrained('facebook/bart-large-mnli')
model_custom = BartForSequenceClassification.from_pretrained(
    'runs/facebook/bart-large/2024-06-13--11-52-22/checkpoints/checkpoint_9', config=config)

# Load the dataset
dataset = load_dataset('glue', 'mnli')
train_dataset = dataset['train']
validation_matched = dataset['validation_matched']
validation_mismatched = dataset['validation_mismatched']


eval_batch_size = 8

def tokenize_function(examples):
    tok_result = tokenizer(examples["premise"], examples["hypothesis"], truncation='only_first', padding="longest", max_length=1024)
    return tok_result

train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=eval_batch_size, remove_columns=["idx", "premise", "hypothesis"]).rename_column("label", "labels")
validation_matched = validation_matched.map(tokenize_function, batched=True, batch_size=eval_batch_size, remove_columns=["idx", "premise", "hypothesis"]).rename_column("label", "labels")
validation_mismatched = validation_mismatched.map(tokenize_function, batched=True, batch_size=eval_batch_size, remove_columns=["idx", "premise", "hypothesis"]).rename_column("label", "labels")



def remap_labels_for_pretrained(examples):
    """
    For some reason, facebook's model reversed the label values
    From their config.json:
      {
        "0": "contradiction",
        "1": "neutral",
        "2": "entailment"
      },

    The mnli dataset uses:
        0 = Entailment
        1 = Neutral
        2 = Contradiction

    As such, we need to remap
    :return: remapped dataset
    """
    remap_dict = {
        0: 2,
        1: 1,
        2: 0,
    }
    examples['labels_remapped'] = [remap_dict[i] for i in examples['labels']]
    return examples


remapped_train_dataset = train_dataset.map(remap_labels_for_pretrained, batched=True, batch_size=eval_batch_size, remove_columns=["labels"]).rename_column("labels_remapped", "labels")
remapped_validation_matched = validation_matched.map(remap_labels_for_pretrained, batched=True, batch_size=eval_batch_size, remove_columns=["labels"]).rename_column("labels_remapped", "labels")
remapped_validation_mismatched = validation_mismatched.map(remap_labels_for_pretrained, batched=True, batch_size=eval_batch_size, remove_columns=["labels"]).rename_column("labels_remapped", "labels")


train_dataset.set_format('torch')
validation_matched.set_format('torch')
validation_mismatched.set_format('torch')

remapped_train_dataset.set_format('torch')
remapped_validation_matched.set_format('torch')
remapped_validation_mismatched.set_format('torch')

def collate_fn(batch):
    default_collated_batch = default_collate(batch)
    correct_device_batch = {k:v.to(device) for k,v in default_collated_batch.items()}
    return correct_device_batch


train_loader = DataLoader(train_dataset, batch_size=eval_batch_size, collate_fn=collate_fn)
validation_matched_loader = DataLoader(validation_matched, batch_size=eval_batch_size, collate_fn=collate_fn)
validation_mismatched_loader = DataLoader(validation_mismatched, batch_size=eval_batch_size, collate_fn=collate_fn)


remapped_train_loader = DataLoader(remapped_train_dataset, batch_size=eval_batch_size, collate_fn=collate_fn)
remapped_validation_matched_loader = DataLoader(remapped_validation_matched, batch_size=eval_batch_size, collate_fn=collate_fn)
remapped_validation_mismatched_loader = DataLoader(remapped_validation_mismatched, batch_size=eval_batch_size, collate_fn=collate_fn)


def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    references = []

    for batch in tqdm.tqdm(dataloader, desc='Evaluating'):
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
accuracy_pretrained_val_matched = evaluate_model(model_pretrained, remapped_validation_matched_loader)
accuracy_custom_val_matched = evaluate_model(model_custom, validation_matched_loader)
accuracy_pretrained_val_mismatched = evaluate_model(model_pretrained, remapped_validation_mismatched_loader)
accuracy_custom_val_mismatched = evaluate_model(model_custom, validation_mismatched_loader)

# Assuming you have the accuracy values stored as mentioned in the previous steps
data = {
    'Model': ['BART-large-MNLI', 'Custom BART'],
    'Validation Matched Accuracy': [accuracy_pretrained_val_matched, accuracy_custom_val_matched],
    'Validation Mismatched Accuracy': [accuracy_pretrained_val_mismatched, accuracy_custom_val_mismatched]
}

if eval_training_data:
    accuracy_pretrained_train = evaluate_model(model_pretrained, remapped_train_loader)
    accuracy_custom_train = evaluate_model(model_custom, train_loader)
    data['Train Accuracy'] = [accuracy_pretrained_train, accuracy_custom_train]

results_df = pd.DataFrame(data)
print(results_df.to_string(), "\n")
