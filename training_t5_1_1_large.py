from training import train_model
from transformers import T5ForSequenceClassification, T5TokenizerFast
import torch

hf_repo = 'google/t5-v1_1-large'

if __name__ == '__main__':

    info_hyperparameters = {
        'hf_repo': hf_repo,
        'precision': 'mixed'
    }

    runs_directory = 'runs'

    # Todo: Replace this with custom tokenizer class, adding a separator token, and replace build_inputs_with_special_tokens
    tokenizer = T5TokenizerFast.from_pretrained(hf_repo)
    max_seq_length = 512

    model = T5ForSequenceClassification.from_pretrained(
        hf_repo,
        num_labels=3,
        # device_map=device,
        # torch_dtype=torch.bfloat16
    )

    tokenizer_kwargs = dict(
        truncation='only_first',
        padding="longest",
        max_length=max_seq_length,
    )

    learning_rate = 1.5e-4

    train_batch_size = 4 # Will be multiplied by device count.

    num_epochs = 5

    num_warmup_steps = (5_000 * 8) // train_batch_size

    train_effective_batch_size = 256

    train_model(tokenizer, model, runs_directory, tokenizer_kwargs, train_batch_size=train_batch_size, learning_rate=learning_rate,
                num_warmup_steps=num_warmup_steps, train_effective_batch_size=train_effective_batch_size, num_epochs=num_epochs,
                info_hyperparameters=info_hyperparameters)
