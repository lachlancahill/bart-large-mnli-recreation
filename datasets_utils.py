from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from config import random_seed
import platform


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
        # also relabel if labels are text.
        'entailment': 2,
        'neutral': 1,
        'contradiction': 0,
    }
    examples[label_col] = [remap_dict[i] for i in examples[label_col]]
    return examples


def perform_remap(input_dataset: DatasetDict, label_col='labels'):
    # remapped_dataset = input_dataset.map(
    #     lambda x: remap_labels_for_consistency(x, label_col=label_col),
    #     batched=True,
    #     batch_size=1000,
    #     num_proc=8,
    # )

    id2label = {
        "0": "contradiction",
        "1": "neutral",
        "2": "entailment",
    }

    label2id = {v:k for k,v in id2label.items()}

    remapped_dataset = input_dataset.align_labels_with_mapping(label2id, label_column=label_col)

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


def get_llama_output_dataset():
    # Original Windows path
    windows_data_path = r'C:\Users\Administrator\PycharmProjects\classification-dataset-generation\balanced_dataset_classify_feedback_using_llama3_result_bank_reviews_combined'

    def convert_path_for_wsl(windows_path):
        """Convert a Windows path to a WSL path."""
        return windows_path.replace('\\', '/').replace('C:', '/mnt/c')

    # Detect if the script is running on WSL
    def is_wsl():
        """Check if the script is running on WSL."""
        if platform.system() == 'Linux':
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    return True
        return False

    # Start with the Windows path
    data_path = windows_data_path

    # Convert the path if running on WSL
    if is_wsl():
        data_path = convert_path_for_wsl(windows_data_path)

    llama_data = load_from_disk(data_path).shuffle(seed=random_seed).train_test_split(test_size=3000, seed=random_seed)

    # Clean and remap in one go
    llama_data = clean_dataset_columns(llama_data)

    # remap to align with original bart_large_mnli model.
    llama_data = perform_remap(llama_data)

    return llama_data


def get_all_datasets():
    public_datasets = get_mnli_anli_snli_combined()

    llama_datasets = get_llama_output_dataset()

    combined_train = concatenate_datasets([
        public_datasets['train'],
        llama_datasets['train']
    ]).shuffle(seed=random_seed)

    # Create a DatasetDict
    combined_dataset = DatasetDict({
        'train': combined_train,
        'mnli_validation_matched': public_datasets['mnli_validation_matched'],
        'mnli_validation_mismatched': public_datasets['mnli_validation_mismatched'],
        'anli_combined_validation': public_datasets['anli_combined_validation'],
        'snli_validation': public_datasets['snli_validation'],
        'llama_labeled_validation': llama_datasets['test'],
    })

    print(f"{combined_dataset=}")

    return combined_dataset


if __name__ == '__main__':
    get_all_datasets()
