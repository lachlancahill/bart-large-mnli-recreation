import datasets_utils
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from training import train_model

if __name__ == '__main__':

    # load base model
    hf_repo = "facebook/bart-large"
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    model = AutoModelForSequenceClassification.from_pretrained(hf_repo, num_labels=3)


    tokenizer_kwargs = dict(
        truncation='only_first',
        padding="longest",
        max_length=tokenizer.model_max_length,
    )

    info_hyperparameters = {
        'hf_repo': hf_repo,
        'precision':'mixed'
    }

    # input_datasets = datasets_utils.get_mnli()
    input_datasets = datasets_utils.get_mnli_anli_snli_combined()

    train_model(
        tokenizer,
        model,
        'runs',
        tokenizer_kwargs,
        input_datasets,
        train_name='train',
        validation_names=None, # figured out by the training function.
        train_effective_batch_size=256,
        train_batch_size=8,
        learning_rate=1.5e-4,
        num_warmup_steps=None,
        num_epochs=3,
        info_hyperparameters=info_hyperparameters,
    )
