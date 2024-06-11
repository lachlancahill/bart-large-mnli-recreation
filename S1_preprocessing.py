import os

from datasets import load_dataset
from config import get_tokenizer, data_path
import compress_pickle as cp

s1_data_path = os.path.join(data_path, 's1_data.pickle.gz')

def preprocess_function(examples):
    # Load BART tokenizer
    tokenizer = get_tokenizer()
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding=True)


# Relabel the dataset to match BART-large-MNLI format
def relabel(example):
    remap_dict = {0: 2, 1: 1, 2: 0}
    example['label'] = remap_dict[example['label']]
    return example
def get_mapped_and_tokenized_dataset():

    # Load MNLI dataset
    mnli = load_dataset('nyu-mll/multi_nli')

    mnli = mnli.map(relabel)

    tokenized_dataset = mnli.map(preprocess_function, batched=True, num_proc=8)

    return mnli, tokenized_dataset

def update_saved_pickle():

    pickle_tuple = get_mapped_and_tokenized_dataset()

    cp.dump(pickle_tuple, s1_data_path)

def get_mapped_and_tokenized_dataset_cached():
    return cp.load(s1_data_path)


if __name__ == '__main__':
    update_saved_pickle()