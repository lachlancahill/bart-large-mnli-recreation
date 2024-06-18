from training import train_model
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast
import torch

hf_repo = 'FacebookAI/xlm-roberta-large'

if __name__ == '__main__':

    info_hyperparameters = {
        'hf_repo': hf_repo,
        'precision': 'mixed'
    }

    runs_directory = 'runs'

    # Todo: Replace this with custom tokenizer class, adding a separator token, and replace build_inputs_with_special_tokens
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(hf_repo)
    max_seq_length = 512

    device = 'cuda'

    model = XLMRobertaForSequenceClassification.from_pretrained(
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

    learning_rate = 1e-4

    train_batch_size = 8

    train_effective_batch_size = 256

    num_epochs = 3


    train_model(tokenizer, model, runs_directory, tokenizer_kwargs,
                train_effective_batch_size=train_effective_batch_size, train_batch_size=train_batch_size,
                learning_rate=learning_rate, num_epochs=num_epochs,
                info_hyperparameters=info_hyperparameters)
