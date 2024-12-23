from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from config import random_seed
import os


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

    label2id = {v: k for k, v in id2label.items()}

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

    remapped_datasets = clean_dataset_columns(raw_datasets)

    remapped_datasets = perform_remap(remapped_datasets)

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


def convert_path_for_wsl(windows_path):
    if not os.name == 'nt':
        print(f"INFO: Converting path `{windows_path}`")
        # Remove drive letter and convert backslashes to forward slashes
        drive, path = windows_path.replace('\\', '/').split(':')
        return f"/mnt/{drive.lower()}{path}"
    else:
        return windows_path


def get_local_dataset(windows_data_path, test_size=0.2):
    data_path = convert_path_for_wsl(windows_data_path)

    llama_data = load_from_disk(data_path).shuffle(seed=random_seed).train_test_split(test_size=test_size,
                                                                                      seed=random_seed)

    # Clean and remap in one go
    llama_data = clean_dataset_columns(llama_data)

    # remap to align with original bart_large_mnli model.
    llama_data = perform_remap(llama_data)

    return llama_data


def get_llama_output_dataset():
    windows_data_path = (r'C:\Users\Administrator\PycharmProjects\classification-'
                         r'dataset-generation\balanced_dataset_classify_feedback_using_llama3_result_bank_'
                         r'reviews_combined')

    return get_local_dataset(windows_data_path)


def get_transcript_context_dataset():
    data_path = r'C:\Users\Administrator\PycharmProjects\sythentic_classification_data\zero_shot_classification_dataset'

    return get_local_dataset(data_path)


def get_transcript_and_mnli():

    mnli = get_mnli()

    alternative_datasets_validation_dict = {}

    synthetic_root = r'C:\Users\Administrator\PycharmProjects\sythentic_classification_data'

    alt_datasets = [# (alt_data_path, validation_data_name)
        # ('zero_shot_classification_dataset', 'llama_transcripts_validation'),
        ('zero_shot_classification_confusing_dataset', 'llama_transcripts_confusing_validation'),
        ('zero_shot_classification_mistral_nemo_dataset', 'mistral_nemo_transcripts_validation'),
        ('zero_shot_classification_qwen_2_5_14b_dataset', 'qwen_2_5_14b_transcripts_validation'),
        # (r'zero_shot_classification_qwen_2_5_coder_7b_dataset', 'qwen_2_5_coder_7b_transcripts_validation'),
    ]

    all_train_datasets = []

    for alt_data_name, validation_data_name in alt_datasets:
        alt_data_path = f"{synthetic_root}\\{alt_data_name}"
        transcripts_dataset_dict = get_local_dataset(alt_data_path)

        print(f"INFO: Adding synthetic data {alt_data_name}:")
        print(f"{transcripts_dataset_dict['train']=}")
        print(f"{transcripts_dataset_dict['test']=}")

        all_train_datasets.append(transcripts_dataset_dict['train'])

        alternative_datasets_validation_dict[validation_data_name] = transcripts_dataset_dict['test']

    combined_train = concatenate_datasets([
        mnli['train'],
        *all_train_datasets,
    ]).shuffle(seed=random_seed)

    # Create a DatasetDict
    combined_dataset = DatasetDict({
        'train': combined_train,
        'mnli_validation_matched': mnli['validation_matched'],
        'mnli_validation_mismatched': mnli['validation_mismatched'],
        **alternative_datasets_validation_dict,
    })

    print(f"{combined_dataset['train']=}")

    return combined_dataset


def get_all_datasets():
    public_datasets = get_mnli_anli_snli_combined()

    llama_datasets = get_llama_output_dataset()

    transcript_dataset = get_transcript_context_dataset()

    combined_train = concatenate_datasets([
        public_datasets['train'],
        llama_datasets['train'],
        transcript_dataset['train'],
    ]).shuffle(seed=random_seed)

    # Create a DatasetDict
    combined_dataset = DatasetDict({
        'train': combined_train,
        'mnli_validation_matched': public_datasets['mnli_validation_matched'],
        'mnli_validation_mismatched': public_datasets['mnli_validation_mismatched'],
        'anli_combined_validation': public_datasets['anli_combined_validation'],
        'snli_validation': public_datasets['snli_validation'],
        'llama_labeled_validation': llama_datasets['test'],
        'llama_transcripts_validation': transcript_dataset['test'],
    })

    print(f"{combined_dataset=}")

    return combined_dataset


if __name__ == '__main__':
    # get_all_datasets()
    get_transcript_and_mnli(mnli_for_evaluation_only=False, include_alternative_datasets=True)
