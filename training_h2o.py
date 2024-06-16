from training import train_model_on_mnli
from transformers import LlamaTokenizerFast, MistralForSequenceClassification

hf_repo = 'h2oai/h2o-danube2-1.8b-base'

if __name__ == '__main__':

    info_hyperparameters = {
        'hf_repo': hf_repo,
        'precision': 'mixed'
    }

    runs_directory = 'runs'

    max_seq_length = 8192

    tokenizer = LlamaTokenizerFast.from_pretrained(hf_repo)

    # configure tokenizer to actually use sequence tokens to delimit premise and hypothesis
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token = True

    model = MistralForSequenceClassification.from_pretrained(hf_repo, num_labels=3)

    if tokenizer.pad_token is None:
        num_added_toks = tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
        assert num_added_toks == 1
        model.resize_token_embeddings(len(tokenizer))
        # model.config.pad_token = tokenizer.pad_token
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer_kwargs = dict(
        truncation='only_first',
        padding="longest",
        max_length=8192,
    )

    learning_rate = 2e-5

    train_batch_size = 2

    num_warmup_steps = (5_000 * 8) // train_batch_size

    gradient_accumulation_steps = 16 // train_batch_size

    train_effective_batch_size = 256

    train_model_on_mnli(tokenizer, model, runs_directory, tokenizer_kwargs, train_batch_size=train_batch_size, learning_rate=learning_rate,
                        num_warmup_steps=num_warmup_steps, train_effective_batch_size=train_effective_batch_size, num_epochs=2,
                        info_hyperparameters=info_hyperparameters)
