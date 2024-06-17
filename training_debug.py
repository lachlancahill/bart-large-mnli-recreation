import datetime
import os.path
import pickle

import torch
from torch.utils.data import DataLoader

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

import evaluate
from accelerate.utils import ProjectConfiguration

raw_datasets = load_dataset("glue", "mnli")

print(f'{raw_datasets=}')

def train_model_on_mnli(tokenizer, model, runs_directory, tokenizer_kwargs, train_effective_batch_size=None,
                        train_batch_size=4, learning_rate=1e-4, num_warmup_steps=None, num_epochs=2,
                        info_hyperparameters=None):

    if train_effective_batch_size is None:
        train_effective_batch_size = train_batch_size

    gradient_accumulation_steps = train_effective_batch_size // train_batch_size

    if num_warmup_steps is None:
        num_warmup_steps = 10000 * gradient_accumulation_steps # ensures warn up steps aligns to effective batch steps

    if info_hyperparameters is None:
        info_hyperparameters = {}

    def get_tensorboard_writer_dir():

        if 'hf_repo' in info_hyperparameters:
            run_dir = f'./{runs_directory}/{info_hyperparameters["hf_repo"]}'
        else:
            run_dir = f'./{runs_directory}'
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        now = datetime.datetime.now()

        now_folder = now.strftime("%Y-%m-%d--%H-%M-%S")

        proj_dir = f"{run_dir}/{now_folder}"

        os.makedirs(proj_dir, exist_ok=True)
        # log_dir = f"{run_dir}/{now_folder}/logs"
        #
        # os.makedirs(log_dir, exist_ok=True)
        #
        # artifact_dir = f"{run_dir}/{now_folder}/artifacts"
        #
        # os.makedirs(artifact_dir, exist_ok=True)

        return proj_dir

    proj_dir = get_tensorboard_writer_dir()

    eval_batch_size = train_batch_size

    def tokenize_function(examples):
        # TODO: Try batch sizes of just 1, with no padding, then just ues gradient accumulation steps. See if the inefficiencies of not using batches are offset by the reduced computation for padded sequences.
        outputs = tokenizer(examples["premise"], examples["hypothesis"], **tokenizer_kwargs)  # TODO: make max length dynamic based on model.
        return outputs

    tokenized_datasets_original_labels = raw_datasets.map(tokenize_function, batched=True,
                                                          # num_proc=8,
                                          remove_columns=["idx", "premise", "hypothesis"], batch_size=train_batch_size)
    tokenized_datasets_original_labels = tokenized_datasets_original_labels.rename_column("label", "labels")


    def remap_labels_for_consistency(examples):
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
            -1:-1, #TODO: these are unlabeled. Need to determine if these can be removed from original datasets.
            0: 2,
            1: 1,
            2: 0,
        }
        examples['labels_remapped'] = [remap_dict[i] for i in examples['labels']]
        return examples

    tokenized_datasets = tokenized_datasets_original_labels.map(remap_labels_for_consistency, batched=True,
                                                                # num_proc=8,
                                                                batch_size=train_batch_size,
                                                                remove_columns=["labels"]
                                                                ).rename_column("labels_remapped", "labels")

    tokenized_datasets.set_format("torch")


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

    # train_dataloader, eval_dataloader, eval_mismatched_dataloader = create_dataloaders()

    metric = evaluate.load("glue", "mnli", trust_remote_code=True)

    hyperparameters = {
        "learning_rate": learning_rate,
        'num_warmup_steps': num_warmup_steps,
        "num_epochs": num_epochs,
        "train_batch_size": train_batch_size,  # Actual batch size will this x devices
        "eval_batch_size": eval_batch_size,  # Actual batch size will this x devices
        'gradient_accumulation_steps': gradient_accumulation_steps,
        "seed": 42,
        **info_hyperparameters,
    }

    def training_function(model):
        # Initialize accelerator

        config = ProjectConfiguration(
            project_dir=proj_dir,
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


        # modify model config so it can be easily reloaded

        id2label = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
        model.config.id2label = id2label
        model.config.label2id = {v: k for k, v in id2label.items()}

        # Save the starting state
        config_dir = f"{proj_dir}/config_checkpoint"
        os.makedirs(config_dir, exist_ok=True)
        model.save_pretrained(config_dir)

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
            num_warmup_steps=hyperparameters['num_warmup_steps'],
            num_training_steps=len(train_dataloader) * num_epochs,
        )

        # Register the scheduler
        # accelerator.register_for_checkpointing(lr_scheduler)


        total_steps = num_epochs * len(train_dataloader)

        # Instantiate a progress bar to keep track of training. Note that we only enable it on the main
        # process to avoid having 8 progress bars.
        progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)

        print(f"INFO: {total_steps=}")

        # The hardcoded numbers are the total number of times through the training run that we want each to happen.
        log_every_x_steps =  total_steps // 1000
        eval_every_x_steps = total_steps // 60
        save_every_x_steps = total_steps // 30

        print(f"INFO: {log_every_x_steps=}")
        print(f"INFO: {eval_every_x_steps=}")
        print(f"INFO: {save_every_x_steps=}")

        def evaluate(model, evaluation_dataloader_arg, dataset_name):
            model.eval()
            all_predictions = []
            all_labels = []

            for step, batch in enumerate(evaluation_dataloader_arg):
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
            accelerator.print(f"{datetime.datetime.now()} epoch {epoch}:", eval_metric)

            # Log the loss as at the gradient accumulation step.
            accelerator.log(eval_metric, step=progress_bar.n)

        # Now we train the model

        train_dataloader_len = len(train_dataloader)
        losses_cache = []

        for epoch in range(num_epochs):
            model.train()

            for step, batch in enumerate(train_dataloader):
                progress_bar.update(1)
                if progress_bar.n < 171805:
                    continue

                try:
                    outputs = model(**batch)
                except Exception as e:
                    # if an error is encountered we want to examine the batch to determine if it is responsible.
                    error_artifacts_dir = f'{proj_dir}/error_artifacts'
                    os.makedirs(error_artifacts_dir, exist_ok=True)
                    with open(os.path.join(error_artifacts_dir, 'error_batch.pickle'), 'wb') as f:
                        pickle.dump(batch, f)
                    raise e

                loss = outputs.loss
                loss = loss / gradient_accumulation_steps

                accelerator.backward(loss)


                master_step_no = progress_bar.n

                step_plus_1 = step + 1

                lr_scheduler.step()

                if step_plus_1 % gradient_accumulation_steps == 0 or step_plus_1 == train_dataloader_len:
                    optimizer.step()
                    optimizer.zero_grad()

                # Append the loss to the list
                losses_cache.append(loss.item())
                if master_step_no % log_every_x_steps == 0:
                    # Log the loss as at the gradient accumulation step.

                    current_lr = float(lr_scheduler.get_last_lr()[0])

                    average_loss = torch.tensor(losses_cache, device=accelerator.device).mean()
                    losses_cache = []  # Clear the cache

                    accelerator.log(
                        {
                            "train_loss": average_loss,
                            'lr': current_lr,
                        },
                        step=progress_bar.n)

                if master_step_no % eval_every_x_steps == 0:
                    evaluate(model, eval_dataloader, 'validation_matched')
                    evaluate(model, eval_mismatched_dataloader, 'validation_mismatched')
                    model.train()

                if master_step_no % save_every_x_steps == 0:
                    accelerator.save_state()

        evaluate(model, eval_dataloader, 'validation_matched')
        evaluate(model, eval_mismatched_dataloader, 'validation_mismatched')

        accelerator.save_state()
        accelerator.end_training()

    training_function(model)


if __name__ == '__main__':
    model_checkpoint = "facebook/bart-large"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

    tokenizer_kwargs = dict(
        truncation='only_first',
        padding="longest",
        max_length=tokenizer.model_max_length,
    )

    info_hyperparameters = {
        'hf_repo': model_checkpoint,
    }

    train_model_on_mnli(
        tokenizer,
        model,
        'runs',
        tokenizer_kwargs,
        train_batch_size=4,
    )
