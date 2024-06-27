import datetime
import os.path
import pickle
from config import random_seed
import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from transformers import (
    get_linear_schedule_with_warmup,
    set_seed,
)
from tqdm.auto import tqdm
from torch.optim import AdamW

import evaluate
from accelerate.utils import ProjectConfiguration, DeepSpeedPlugin, DummyOptim, DummyScheduler


def train_model(tokenizer, model, runs_directory, tokenizer_kwargs, input_datasets, train_name='train',
                validation_names=None, train_effective_batch_size=256, train_batch_size=4, learning_rate=1e-4,
                num_warmup_steps=None, num_epochs=2, info_hyperparameters=None, checkpoint_dir=None,
                add_llm_framing=False, custom_llm_framing_function=None):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    if checkpoint_dir is not None:
        info_hyperparameters['checkpoint_dir'] = checkpoint_dir

    def count_trainable_parameters(model_passed):
        return sum(p.numel() for p in model_passed.parameters() if p.requires_grad)

    info_hyperparameters['total_trainable_params'] = count_trainable_parameters(model)

    if validation_names is None:
        validation_names = [c for c in input_datasets.keys() if c != 'train']
        print(f"INFO: {validation_names=}")

    gradient_accumulation_steps = train_effective_batch_size // train_batch_size

    if num_warmup_steps is None:
        total_steps = len(input_datasets[train_name]) // train_batch_size
        num_warmup_steps = total_steps // 10  # 1/10th of the total steps should be spent on warm up.

    if info_hyperparameters is None:
        info_hyperparameters = {}

    def get_tensorboard_writer_dir():

        if 'hf_repo' in info_hyperparameters:
            run_dir = f'./{runs_directory}/{info_hyperparameters["hf_repo"]}'
        else:
            run_dir = f'./{runs_directory}'
        os.makedirs(run_dir, exist_ok=True)

        now = datetime.datetime.now()

        now_folder = now.strftime("%Y-%m-%d--%H-%M")

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

    def add_framing_for_llm_classification_default(premises, hypotheses):

        zsc_prefix = 'The following is a Zero Shot Classification Problem.\nPremise:\n'
        zsc_hypothesis = '\nHypothesis:\n'
        zsc_suffix = '\nClassification (Entailment, Neutral or Contradiction):\n'

        new_premises, new_hypotheses = [], []
        for premise, hypothesis in zip(premises, hypotheses):
            new_premises.append(f'{zsc_prefix}{premise}')
            new_hypotheses.append(f'{zsc_hypothesis}{hypothesis}{zsc_suffix}')

        return new_premises, new_hypotheses

    framing_function_to_use = add_framing_for_llm_classification_default if custom_llm_framing_function is None else custom_llm_framing_function

    def tokenize_function(examples):

        if add_llm_framing:
            new_premises, new_hypotheses = framing_function_to_use(examples["premise"], examples["hypothesis"])
            outputs = tokenizer(new_premises, new_hypotheses, **tokenizer_kwargs)
        else:
            outputs = tokenizer(examples["premise"], examples["hypothesis"],**tokenizer_kwargs)

        return outputs

    tokenized_datasets = input_datasets.map(tokenize_function, batched=True,
                                            # num_proc=8,
                                            remove_columns=["premise", "hypothesis"], batch_size=train_batch_size)

    tokenized_datasets.set_format("torch")

    def create_dataloaders(train_batch_size=train_batch_size, eval_batch_size=eval_batch_size):
        train_dataloader = DataLoader(
            tokenized_datasets[train_name], shuffle=False, batch_size=train_batch_size, drop_last=True
        )

        eval_dataloader_dict = {}

        for eval_name in validation_names:
            eval_dataloader_dict[eval_name] = DataLoader(
                tokenized_datasets[eval_name], shuffle=False, batch_size=eval_batch_size, drop_last=True
            )
        return train_dataloader, eval_dataloader_dict

    # train_dataloader, eval_dataloader, eval_mismatched_dataloader = create_dataloaders()

    metric = evaluate.load("glue", "mnli", trust_remote_code=True)

    hyperparameters = {
        "learning_rate": learning_rate,
        'num_warmup_steps': num_warmup_steps,
        "num_epochs": num_epochs,
        "train_batch_size": train_batch_size,  # Actual batch size will this x devices
        "eval_batch_size": eval_batch_size,  # Actual batch size will this x devices
        'gradient_accumulation_steps': gradient_accumulation_steps,
        "seed": random_seed,
        **info_hyperparameters,
    }

    def training_function(model):
        # Initialize accelerator

        config = ProjectConfiguration(
            project_dir=proj_dir,
            automatic_checkpoint_naming=True
        )

        deepspeed_plugin = DeepSpeedPlugin(
            # hf_ds_config='./zero_stage_3_config.json',
            gradient_accumulation_steps=hyperparameters['gradient_accumulation_steps'],
            gradient_clipping=True,
            zero_stage=2,
            offload_optimizer_device='none',
            offload_param_device='none',
            zero3_init_flag=False,
            zero3_save_16bit_model=False,
        )



        accelerator = Accelerator(
            log_with="tensorboard",
            project_config=config,
            gradient_accumulation_steps=hyperparameters['gradient_accumulation_steps'],
            # deepspeed_plugin=deepspeed_plugin,
            mixed_precision='bf16',
            dataloader_config=None,

            # split_batches=True
        )

        accelerator.init_trackers("logs", config=hyperparameters)

        # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
        # to INFO for the main process only.
        # if accelerator.is_main_process:
        #     datasets.utils.logging.set_verbosity_warning()
        #     transformers.utils.logging.set_verbosity_info()
        # else:
        #     datasets.utils.logging.set_verbosity_error()
        #     transformers.utils.logging.set_verbosity_error()

        train_dataloader, eval_dataloader_dict = create_dataloaders(
            train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
        )
        # The seed need to be set before we instantiate the model, as it will determine the random head.
        set_seed(hyperparameters["seed"])

        print(f"{accelerator.num_processes=}")

        # Instantiate optimizer

        optimizer_cls = (
            AdamW
            if accelerator.state.deepspeed_plugin is None
               or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
        )

        optimizer = optimizer_cls(params=model.parameters(), lr=hyperparameters["learning_rate"])

        # modify model config so it can be easily reloaded

        id2label = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
        # TODO: Address the warning saying this needs to be in a generation config.
        model.config.id2label = id2label
        model.config.label2id = {v: k for k, v in id2label.items()}

        # Save the starting state
        config_dir = f"{proj_dir}/config_checkpoint"
        os.makedirs(config_dir, exist_ok=True)
        model.save_pretrained(config_dir)

        num_epochs = hyperparameters["num_epochs"]
        gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
        # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
        # may change its length.

        # Prepare everything
        # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
        # prepare method.
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

        if (
                accelerator.state.deepspeed_plugin is None
                or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=hyperparameters['num_warmup_steps'],
                num_training_steps=len(train_dataloader) * num_epochs,
            )
        else:
            lr_scheduler = DummyScheduler(
                optimizer,
                num_warmup_steps=hyperparameters['num_warmup_steps'],
                num_training_steps=len(train_dataloader) * num_epochs,
            )

        lr_scheduler = accelerator.prepare(lr_scheduler)

        for eval_name in eval_dataloader_dict.keys():
            eval_dataloader_dict[eval_name] = accelerator.prepare(eval_dataloader_dict[eval_name])

        if checkpoint_dir is not None:
            model.to('cpu') # Move to cpu while loading states to avoid potential memory errors.
            accelerator.load_state(checkpoint_dir)
            model.to(accelerator.device)

        # Register the scheduler
        # accelerator.register_for_checkpointing(lr_scheduler)
        total_steps = num_epochs * len(train_dataloader)

        # Instantiate a progress bar to keep track of training. Note that we only enable it on the main
        # process to avoid having 8 progress bars.
        # progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)

        print(f"{accelerator.is_local_main_process=}")
        print(f"{accelerator.is_main_process=}")

        # progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
        progress_bar = tqdm(range(total_steps), disable=False)

        print(f"INFO: {total_steps=}")

        # The hardcoded numbers are the total number of times through the training run that we want each to happen.
        log_every_x_steps = 256
        eval_every_x_steps = total_steps // 60
        save_every_x_steps = total_steps // 30

        print(f"INFO: {log_every_x_steps=}")
        print(f"INFO: {eval_every_x_steps=}")
        print(f"INFO: {save_every_x_steps=}")

        error_artifacts_dir = f'{proj_dir}/error_artifacts'

        def evaluate_checkpoint(model, evaluation_dataloader_arg, dataset_name):
            all_predictions = []
            all_labels = []

            for step, batch in enumerate(evaluation_dataloader_arg):
                try:
                    with torch.no_grad():
                        outputs = model(**batch)
                except Exception as e:
                    # if an error is encountered we want to examine the batch to determine if it is responsible.
                    os.makedirs(error_artifacts_dir, exist_ok=True)
                    with open(os.path.join(error_artifacts_dir, 'error_batch.pickle'), 'wb') as f:
                        pickle.dump(batch, f)
                    raise e

                predictions = outputs.logits.argmax(dim=-1)

                # We gather predictions and labels from the GPUs.
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

        # TODO: First train just the classification head by freezing all other parameters, then unfreeze and
        #  train the full model.

        for epoch in range(num_epochs):
            model.train()

            for step, batch in enumerate(train_dataloader):

                with accelerator.accumulate(model):

                    try:
                        outputs = model(**batch)
                    except Exception as e:
                        # if an error is encountered we want to examine the batch to determine if it is responsible.
                        os.makedirs(error_artifacts_dir, exist_ok=True)
                        with open(os.path.join(error_artifacts_dir, 'error_batch.pickle'), 'wb') as f:
                            pickle.dump(batch, f)
                        raise e

                    loss_raw = outputs.loss
                    loss = loss_raw / gradient_accumulation_steps
                    # loss = loss_raw

                    accelerator.backward(loss)

                    lr_scheduler.step()

                    # if master_step_no % gradient_accumulation_steps == 0 or master_step_no == train_dataloader_len:
                    optimizer.step()
                    optimizer.zero_grad()

                # print(f"{accelerator.is_main_process=}")

                progress_bar.update(1)

                master_step_no = progress_bar.n

                raw_loss_item = loss_raw.item()

                # print(f"raw loss item: {raw_loss_item}")

                # Append the loss to the list
                losses_cache.append(raw_loss_item)
                if master_step_no % log_every_x_steps == 0:
                    # Log the loss as at the gradient accumulation step.

                    current_lr = float(lr_scheduler.get_last_lr()[0])

                    average_loss = torch.tensor(losses_cache, device='cpu').mean().item()
                    losses_cache = []  # Clear the cache

                    log_data = {
                            "train_loss": average_loss,
                            'lr': current_lr,
                        }
                    # print(f"Logging: {log_data}")

                    # print(f"{accelerator.is_main_process}")

                    accelerator.log(
                        log_data,
                        step=progress_bar.n)

                if master_step_no % eval_every_x_steps == 0:
                    model.eval()
                    for eval_name in eval_dataloader_dict.keys():
                        evaluate_checkpoint(model, eval_dataloader_dict[eval_name], eval_name)
                    model.train()

                if master_step_no % save_every_x_steps == 0:
                    accelerator.wait_for_everyone()
                    accelerator.save_state()

        model.eval()
        # final evaluation completed.
        for eval_name in eval_dataloader_dict.keys():
            evaluate_checkpoint(model, eval_dataloader_dict[eval_name], eval_name)

        accelerator.wait_for_everyone()
        accelerator.save_state()
        accelerator.end_training()

    training_function(model)
