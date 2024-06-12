import datetime
import os.path

import torch
from torch.utils.data import DataLoader, Subset

from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from tqdm.auto import tqdm
from torch.optim import AdamW

from config import model_checkpoint

import evaluate
from accelerate.utils import ProjectConfiguration

raw_datasets = load_dataset("glue", "mnli")

print(f'{raw_datasets=}')

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def get_tensorboard_writer_dir():
    run_dir = './runs'
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    now = datetime.datetime.now()

    now_folder = now.strftime("%Y-%m-%d--%H-%M-%S")

    proj_dir = f"{run_dir}/{now_folder}"

    os.makedirs( proj_dir, exist_ok=True)
    # log_dir = f"{run_dir}/{now_folder}/logs"
    #
    # os.makedirs(log_dir, exist_ok=True)
    #
    # artifact_dir = f"{run_dir}/{now_folder}/artifacts"
    #
    # os.makedirs(artifact_dir, exist_ok=True)

    return proj_dir


proj_dir = get_tensorboard_writer_dir()


def tokenize_function(examples):
    outputs = tokenizer(examples["premise"], examples["hypothesis"], truncation='only_first', padding="max_length",
                        max_length=1024)  # TODO: make max length dynamic.
    return outputs


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["idx", "premise", "hypothesis"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

train_batch_size = 8
eval_batch_size = 64
eval_every_x_steps = 20_000
save_every_x_steps = eval_every_x_steps

def create_dataloaders(train_batch_size=train_batch_size, eval_batch_size=eval_batch_size):
    train_dataloader = DataLoader(
        # # TODO: Remove filter on training data.
        # Subset(tokenized_datasets["train"], range(1_000)), shuffle=False, batch_size=train_batch_size
        tokenized_datasets["train"], shuffle=False, batch_size=train_batch_size
    )
    eval_matched_dataloader = DataLoader(
        tokenized_datasets["validation_matched"], shuffle=False, batch_size=eval_batch_size
    )
    eval_mismatched_dataloader = DataLoader(
        tokenized_datasets["validation_mismatched"], shuffle=False, batch_size=eval_batch_size
    )
    return train_dataloader, eval_matched_dataloader, eval_mismatched_dataloader


train_dataloader, eval_dataloader, eval_mismatched_dataloader = create_dataloaders()

# for batch in train_dataloader:
#     print({k: v.shape for k, v in batch.items()})
#     outputs = model(**batch)
#     break


metric = evaluate.load("glue", "mnli", trust_remote_code=True)

# predictions = outputs.logits.detach().argmax(dim=-1)
# metric.compute(predictions=predictions, references=batch["labels"])

hyperparameters = {
    "learning_rate": 2e-5,
    "num_epochs": 2,
    "train_batch_size": train_batch_size,  # Actual batch size will this x 8
    "eval_batch_size": eval_batch_size,  # Actual batch size will this x 8
    'gradient_accumulation_steps': 2,
    "seed": 42,
}


def training_function(model):
    # Initialize accelerator

    config = ProjectConfiguration(
        project_dir=proj_dir,
        # logging_dir=proj_dir,
        automatic_checkpoint_naming=True
    )
    accelerator = Accelerator(log_with="tensorboard", project_config=config)

    accelerator.init_trackers("logs", config=hyperparameters)

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    # if accelerator.is_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()

    train_dataloader, eval_dataloader, eval_mismatched_dataloader = create_dataloaders(
        train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
    )
    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(hyperparameters["seed"])

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=hyperparameters["learning_rate"])

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, eval_mismatched_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, eval_mismatched_dataloader
    )

    num_epochs = hyperparameters["num_epochs"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) // gradient_accumulation_steps) * num_epochs,
    )

    # Register the scheduler
    # accelerator.register_for_checkpointing(lr_scheduler)

    # Save the starting state
    accelerator.save_state()

    # Instantiate a progress bar to keep track of training. Note that we only enable it on the main
    # process to avoid having 8 progress bars.
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)), disable=not accelerator.is_main_process)

    def evaluate(model, evaluation_dataloader_arg, dataset_name):
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
        all_predictions = torch.cat(all_predictions)[:len(tokenized_datasets[dataset_name])]
        all_labels = torch.cat(all_labels)[:len(tokenized_datasets[dataset_name])]

        eval_metric = metric.compute(predictions=all_predictions, references=all_labels)

        eval_metric = {f"{k}_{dataset_name}": v for k, v in eval_metric.items()}

        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)

        # Log the loss as at the gradient accumulation step.
        accelerator.log(eval_metric, step=progress_bar.n)

    # Now we train the model

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps

            accelerator.backward(loss)

            if (step + 1) % gradient_accumulation_steps == 0:

                # Log the loss as at the gradient accumulation step.
                accelerator.log({"train_loss": loss}, step=progress_bar.n)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (progress_bar.n + 1) % eval_every_x_steps == 0:
                evaluate(model, eval_dataloader, 'validation_matched')
                evaluate(model, eval_mismatched_dataloader, 'validation_mismatched')
                model.train()


            if (progress_bar.n + 1) % save_every_x_steps == 0:
                accelerator.save_state()

            progress_bar.update(1)


    evaluate(model, eval_dataloader, 'validation_matched')
    evaluate(model, eval_mismatched_dataloader, 'validation_mismatched')

    accelerator.save_state()
    accelerator.end_training()


training_function(model)

# from accelerate import notebook_launcher
#
# notebook_launcher(training_function, (model,), num_processes=2)
