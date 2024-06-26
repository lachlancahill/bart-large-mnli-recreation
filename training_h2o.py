from training import train_model
from transformers import LlamaTokenizerFast, MistralForSequenceClassification, AutoConfig
import datasets_utils

hf_repo = 'h2oai/h2o-danube2-1.8b-base'

if __name__ == '__main__':

    info_hyperparameters = {
        'hf_repo': hf_repo,
        'precision': 'mixed'
    }

    runs_directory = 'runs'

    max_seq_length = 2048

    tokenizer = LlamaTokenizerFast.from_pretrained(hf_repo)

    # configure tokenizer to actually use sequence tokens to delimit premise and hypothesis
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token = True

    run_of_interest = './runs/h2oai/h2o-danube2-1.8b-base/2024-06-21--18-54-33'

    checkpoint_dir = rf'{run_of_interest}/checkpoints/checkpoint_4'

    # config = AutoConfig.from_pretrained(f'{run_of_interest}/config_checkpoint')

    model = MistralForSequenceClassification.from_pretrained(hf_repo, num_labels=3)

    if tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
        assert num_added_tokens == 1
        model.resize_token_embeddings(len(tokenizer))
        # model.config.pad_token = tokenizer.pad_token
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer_kwargs = dict(
        truncation='only_first',
        padding="longest",
        max_length=max_seq_length,
    )

    # TODO: freeze earliest parameters to keep word meanings from original model.

    learning_rate = 1e-5

    train_batch_size = 2

    # input_datasets = datasets_utils.get_mnli()
    input_datasets = datasets_utils.get_mnli_anli_snli_combined()

    # checkpoint_dir = None

    train_model(
        tokenizer,
        model,
        'runs',
        tokenizer_kwargs,
        input_datasets,
        train_name='train',
        validation_names=None, # figured out by the training function.
        train_effective_batch_size=256,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        num_warmup_steps=None,
        num_epochs=7,
        info_hyperparameters=info_hyperparameters,
        checkpoint_dir=checkpoint_dir,
    )

