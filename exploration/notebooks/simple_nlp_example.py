# ! pip install datasets transformers evaluate
# ! pip install cloud-tpu-client==0.10 torch==2.0.0
# ! pip install https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl
# ! pip install git+https://github.com/huggingface/accelerate

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType
from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from tqdm.auto import tqdm

import datasets
import transformers

model_checkpoint = "bert-base-cased"

raw_datasets = load_dataset("glue", "mrpc")

raw_datasets

raw_datasets["train"][0]

import datasets
# import random
# import pandas as pd
# from IPython.display import display, HTML

# def show_random_elements(dataset, num_examples=10):
#     assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
#     picks = []
#     for _ in range(num_examples):
#         pick = random.randint(0, len(dataset)-1)
#         while pick in picks:
#             pick = random.randint(0, len(dataset)-1)
#         picks.append(pick)
#
#     df = pd.DataFrame(dataset[picks])
#     for column, typ in dataset.features.items():
#         if isinstance(typ, datasets.ClassLabel):
#             df[column] = df[column].transform(lambda i: typ.names[i])
#     display(HTML(df.to_html()))
#
# show_random_elements(raw_datasets["train"])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenizer("Hello, this one sentence!", "And this sentence goes with it.")

def tokenize_function(examples):
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)
    return outputs

tokenize_function(raw_datasets['train'][:5])

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["idx", "sentence1", "sentence2"])

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

tokenized_datasets["train"].features

tokenized_datasets.set_format("torch")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

def create_dataloaders(train_batch_size=8, eval_batch_size=32):
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=train_batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, batch_size=eval_batch_size
    )
    return train_dataloader, eval_dataloader

train_dataloader, eval_dataloader = create_dataloaders()

for batch in train_dataloader:
    print({k: v.shape for k, v in batch.items()})
    outputs = model(**batch)
    break

outputs

import evaluate
metric = evaluate.load("glue", "mrpc", trust_remote_code=True)

predictions = outputs.logits.detach().argmax(dim=-1)
metric.compute(predictions=predictions, references=batch["labels"])

hyperparameters = {
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "train_batch_size": 8, # Actual batch size will this x 8
    "eval_batch_size": 32, # Actual batch size will this x 8
    "seed": 42,
}

def training_function(model):
    # Initialize accelerator
    accelerator = Accelerator()

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    train_dataloader, eval_dataloader = create_dataloaders(
        train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
    )
    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(hyperparameters["seed"])

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=hyperparameters["learning_rate"])

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_epochs = hyperparameters["num_epochs"]
    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    # Instantiate a progress bar to keep track of training. Note that we only enable it on the main
    # process to avoid having 8 progress bars.
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)), disable=not accelerator.is_main_process)
    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        all_predictions = []
        all_labels = []

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            # We gather predictions and labels from the 8 TPUs to have them all.
            all_predictions.append(accelerator.gather(predictions))
            all_labels.append(accelerator.gather(batch["labels"]))

        # Concatenate all predictions and labels.
        # The last thing we need to do is to truncate the predictions and labels we concatenated
        # together as the prepared evaluation dataloader has a little bit more elements to make
        # batches of the same size on each process.
        all_predictions = torch.cat(all_predictions)[:len(tokenized_datasets["validation"])]
        all_labels = torch.cat(all_labels)[:len(tokenized_datasets["validation"])]

        eval_metric = metric.compute(predictions=all_predictions, references=all_labels)

        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)

training_function(model)

# from accelerate import notebook_launcher
#
# notebook_launcher(training_function, (model,), num_processes=2)


