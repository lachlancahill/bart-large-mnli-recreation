from training import train_model_on_mnli
from transformers import T5ForSequenceClassification, T5TokenizerFast
import torch

hf_repo = 'google-t5/t5-large'

if __name__ == '__main__':

    info_hyperparameters = {
        'hf_repo': hf_repo,
        'precision': 'bfloat16'
    }

    runs_directory = 'runs'


    tokenizer = T5TokenizerFast.from_pretrained(hf_repo)
    max_seq_length = 2048

    device = 'cuda'

    model = T5ForSequenceClassification.from_pretrained(
        hf_repo,
        num_labels=3,
        device_map=device,
        torch_dtype=torch.bfloat16
    )

    tokenizer_kwargs = dict(
        truncation='only_first',
        padding="longest",
        max_length=max_seq_length,
    )

    learning_rate = 3e-5

    train_batch_size = 8

    num_epochs = 3

    num_warmup_steps = (5_000 * 8) // train_batch_size

    gradient_accumulation_steps = 16 // train_batch_size

    train_model_on_mnli(tokenizer, model, runs_directory, tokenizer_kwargs, train_batch_size=train_batch_size, learning_rate=learning_rate,
                        num_warmup_steps=num_warmup_steps, gradient_accumulation_steps=gradient_accumulation_steps, num_epochs=num_epochs,
                        info_hyperparameters=info_hyperparameters)
