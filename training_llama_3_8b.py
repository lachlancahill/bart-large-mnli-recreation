from training import train_model
from transformers import PreTrainedTokenizerFast, LlamaForSequenceClassification, AutoConfig
import datasets_utils

hf_repo = 'meta-llama/Meta-Llama-3-8B-Instruct'


def llama_3_framing_function(premises, hypotheses):

    zsc_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

The user will provide a Zero Shot Classification Problem. Respond only with the correct classification 'Entailment', 'Neutral' or 'Contradiction'.<|eot_id|><|start_header_id|>user<|end_header_id|>

Premise:
{premise}

Hypothesis:
{hypothesis}

Respond with only one word, the correct classification 'Entailment', 'Neutral' or 'Contradiction'.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
The correct classification is '"""


    new_premises, new_hypotheses = [], []
    for premise, hypothesis in zip(premises, hypotheses):
        new_premises.append(zsc_template.format(premise=premise, hypothesis=hypothesis))
        # new_hypotheses.append(f'{zsc_hypothesis}{hypothesis}{zsc_suffix}')
        new_hypotheses.append('') # remove as all information is in the premise.

    return new_premises, new_hypotheses


if __name__ == '__main__':


    runs_directory = 'runs'

    max_seq_length = 8000

    tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_repo)

    # configure tokenizer to actually use sequence tokens to delimit premise and hypothesis
    # tokenizer.add_eos_token = True
    # tokenizer.add_bos_token = True

    # run_of_interest = './runs/h2oai/h2o-danube2-1.8b-base/2024-06-21--18-54-33'
    # checkpoint_dir = rf'{run_of_interest}/checkpoints/checkpoint_4'

    # config = AutoConfig.from_pretrained(f'{run_of_interest}/config_checkpoint')

    model = LlamaForSequenceClassification.from_pretrained(hf_repo, num_labels=3)
    assert tokenizer.eos_token is not None and model.config.eos_token_id is not None
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


    tokenizer_kwargs = dict(
        truncation='only_first',
        padding="longest",
        max_length=max_seq_length,
    )

    # Fine-tune only the last block and final normalization layer
    for param in model.parameters():
        param.requires_grad = False

    print(f"INFO: Total number of layers: {len(model.model.layers)}")

    num_layers_to_unfreeze = 3
    # Fine-tune only the final normalization layer
    for block in model.model.layers[-num_layers_to_unfreeze:]:
        for param in block.parameters():
            param.requires_grad = True

    final_norm = model.model.norm
    for param in final_norm.parameters():
        param.requires_grad = True


    info_hyperparameters = {
        'hf_repo': hf_repo,
        'precision': 'mixed',
        'add_llm_framing': 'True',
        'num_layers_to_unfreeze':num_layers_to_unfreeze,
    }

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
        add_llm_framing=True,
        custom_llm_framing_function=llama_3_framing_function
        # checkpoint_dir=checkpoint_dir,
    )

