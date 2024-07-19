import tqdm
from transformers import BartForSequenceClassification, BartTokenizer, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data.dataloader import default_collate


eval_training_data = True

# parameters relating to the custom model we are testing.
original_id = 'facebook/bart-large-mnli'
config_path = r'C:\Users\Administrator\PycharmProjects\bart-large-mnli-recreation\runs\facebook\bart-large\2024-07-17--23-09\config_checkpoint'
checkpoint_path = r'C:\Users\Administrator\PycharmProjects\bart-large-mnli-recreation\runs\facebook\bart-large\2024-07-17--23-09\checkpoints\checkpoint_30'
max_length = 1024


device = torch.device('cuda:0')

# Load Facebook's pre-trained BART-large-MNLI model
bart_id = 'facebook/bart-large-mnli'
tokenizer_pretrained = BartTokenizer.from_pretrained(bart_id)
model_pretrained = BartForSequenceClassification.from_pretrained(bart_id, device_map=device, torch_dtype=torch.bfloat16)

# Load your custom-trained model
config = AutoConfig.from_pretrained(config_path)
tokenizer_custom = AutoTokenizer.from_pretrained(original_id)
model_custom = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, config=config, device_map=device, torch_dtype=torch.bfloat16)

# Load the dataset
from datasets_utils import get_llama_output_dataset
dataset = get_llama_output_dataset()
train_dataset = dataset['train']
validation_matched = dataset['test']
# validation_mismatched = dataset['validation_mismatched']


eval_batch_size = 8

def tokenize_function_pretrained(examples):
    tok_result = tokenizer_pretrained(examples["premise"], examples["hypothesis"], truncation='only_first', padding="longest", max_length=1024)
    return tok_result


def tokenize_function_custom(examples):
    tok_result = tokenizer_custom(examples["premise"], examples["hypothesis"], truncation='only_first', padding="longest", max_length=max_length)
    return tok_result


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
    examples['labels'] = [remap_dict[i] for i in examples['label']]
    return examples


# remapped_train_dataset = train_dataset.map(remap_labels_for_pretrained, batched=True, batch_size=eval_batch_size, remove_columns=["label"])
# remapped_validation_matched = validation_matched.map(remap_labels_for_pretrained, batched=True, batch_size=eval_batch_size, remove_columns=["label"])
# remapped_validation_mismatched = validation_mismatched.map(remap_labels_for_pretrained, batched=True, batch_size=eval_batch_size, remove_columns=["label"])


train_dataset_pretrained = train_dataset.map(tokenize_function_pretrained, batched=True, batch_size=eval_batch_size, remove_columns=["premise", "hypothesis"])
validation_matched_pretrained = validation_matched.map(tokenize_function_pretrained, batched=True, batch_size=eval_batch_size, remove_columns=["premise", "hypothesis"])
# validation_mismatched_pretrained = remapped_validation_mismatched.map(tokenize_function_pretrained, batched=True, batch_size=eval_batch_size, remove_columns=["idx", "premise", "hypothesis"])

train_dataset_custom = train_dataset.map(tokenize_function_custom, batched=True, batch_size=eval_batch_size, remove_columns=["premise", "hypothesis"])
validation_matched_custom = validation_matched.map(tokenize_function_custom, batched=True, batch_size=eval_batch_size, remove_columns=["premise", "hypothesis"])
# validation_mismatched_custom = remapped_validation_mismatched.map(tokenize_function_custom, batched=True, batch_size=eval_batch_size, remove_columns=["idx", "premise", "hypothesis"])


train_dataset_pretrained.set_format('torch')
validation_matched_pretrained.set_format('torch')
# validation_mismatched_pretrained.set_format('torch')

train_dataset_custom.set_format('torch')
validation_matched_custom.set_format('torch')
# validation_mismatched_custom.set_format('torch')

def collate_fn(batch):
    default_collated_batch = default_collate(batch)
    correct_device_batch = {k:v.to(device) for k,v in default_collated_batch.items()}
    return correct_device_batch


loader_train_dataset_pretrained = DataLoader(train_dataset_pretrained, batch_size=eval_batch_size, collate_fn=collate_fn)
loader_validation_matched_pretrained = DataLoader(validation_matched_pretrained, batch_size=eval_batch_size, collate_fn=collate_fn)
# loader_validation_mismatched_pretrained = DataLoader(validation_mismatched_pretrained, batch_size=eval_batch_size, collate_fn=collate_fn)


loader_train_dataset_custom = DataLoader(train_dataset_custom, batch_size=eval_batch_size, collate_fn=collate_fn)
loader_validation_matched_custom = DataLoader(validation_matched_custom, batch_size=eval_batch_size, collate_fn=collate_fn)
# loader_validation_mismatched_custom = DataLoader(validation_mismatched_custom, batch_size=eval_batch_size, collate_fn=collate_fn)


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

# Evaluate on different splits
accuracy_pretrained_val_matched = evaluate_model(model_pretrained, loader_validation_matched_pretrained)
accuracy_custom_val_matched = evaluate_model(model_custom, loader_validation_matched_custom)
# accuracy_pretrained_val_mismatched = evaluate_model(model_pretrained, loader_validation_mismatched_pretrained)
# accuracy_custom_val_mismatched = evaluate_model(model_custom, loader_validation_mismatched_custom)

# Assuming you have the accuracy values stored as mentioned in the previous steps
data = {
    'Model': ['BART-large-MNLI', 'Custom Model'],
    'Validation Matched Accuracy': [accuracy_pretrained_val_matched, accuracy_custom_val_matched],
    # 'Validation Mismatched Accuracy': [accuracy_pretrained_val_mismatched, accuracy_custom_val_mismatched]
}

if eval_training_data:
    accuracy_pretrained_train = evaluate_model(model_pretrained, loader_train_dataset_pretrained)
    accuracy_custom_train = evaluate_model(model_custom, loader_train_dataset_custom)
    data['Train Accuracy'] = [accuracy_pretrained_train, accuracy_custom_train]

results_df = pd.DataFrame(data)
print(results_df.to_string(), "\n")

