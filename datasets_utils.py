from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from config import random_seed


def remap_labels_for_consistency(examples, label_col='labels'):
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
        # -1: -1,  # TODO: these are unlabeled. Need to determine if these can be removed from original datasets.
        0: 2,
        1: 1,
        2: 0,
    }
    examples[label_col] = [remap_dict[i] for i in examples[label_col]]
    return examples


def perform_remap(input_dataset: DatasetDict, label_col='labels'):
    remapped_dataset = input_dataset.map(
        lambda x: remap_labels_for_consistency(x, label_col=label_col),
        batched=True,
        batch_size=1000,
        num_proc=8,
    )

    return remapped_dataset


def clean_dataset_columns(input_dataset_dict: DatasetDict):
    rename = {
        'label': 'labels'
    }

    keep = ['premise', 'hypothesis', 'labels']

    for dataset_name, dataset in input_dataset_dict.items():
        # Rename columns
        for old_name, new_name in rename.items():
            if old_name in dataset.column_names:
                dataset = dataset.rename_column(old_name, new_name)

        # Remove columns not in 'keep'
        columns_to_remove = [col for col in dataset.column_names if col not in keep]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

        # Update the dataset in the DatasetDict
        input_dataset_dict[dataset_name] = dataset

    # remove unlabelled entries.
    input_dataset_dict = input_dataset_dict.filter(lambda x: x['labels'] != -1)

    return input_dataset_dict


def get_mnli():
    raw_datasets = load_dataset("glue", "mnli")
    raw_datasets = raw_datasets.filter(function=lambda x: x['label'] != -1)

    remapped_datasets = perform_remap(raw_datasets)

    remapped_datasets = clean_dataset_columns(remapped_datasets)

    return remapped_datasets


def get_mnli_anli_snli_combined():
    # Load the datasets
    mnli = load_dataset('glue', 'mnli')  # Needs remapping
    anli = load_dataset('facebook/anli')  # Needs remapping
    snli = load_dataset('stanfordnlp/snli')  # Needs remapping

    # clean columns to ensure all datasets have the same columns
    mnli = clean_dataset_columns(mnli)
    anli = clean_dataset_columns(anli)
    snli = clean_dataset_columns(snli)

    # remap to align with original bart_large_mnli model.
    mnli = perform_remap(mnli)
    anli = perform_remap(anli)
    snli = perform_remap(snli)


    # Combine the train datasets
    combined_train = concatenate_datasets([
        # mnli
        mnli['train'], mnli['test_matched'],  # don't include test mismatched

        # anli
        anli['train_r1'], anli['train_r2'], anli['train_r3'],
        anli['dev_r1'], anli['dev_r2'], anli['dev_r3'],

        # snli
        snli['train'], snli['test']
    ]).shuffle(seed=random_seed)  # shuffle so we get a good mix of the different datasets as we go.

    anli_combined_validation = concatenate_datasets([
        anli['test_r1'], anli['test_r2'], anli['test_r3'],
    ])

    # Create a DatasetDict
    combined_dataset = DatasetDict({
        'train': combined_train,
        'mnli_validation_matched': mnli['validation_matched'],
        'mnli_validation_mismatched': mnli['validation_mismatched'],
        'anli_combined_validation': anli_combined_validation,
        'snli_validation': snli['validation'],
    })

    print(f"{combined_dataset=}")

    return combined_dataset