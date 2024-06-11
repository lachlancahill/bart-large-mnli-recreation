import os.path

import torch
from transformers import BartTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_checkpoint = "facebook/bart-large"

data_path = './data'

if not os.path.exists(data_path):
    os.makedirs(data_path)

def get_tokenizer():
    return BartTokenizer.from_pretrained(model_checkpoint)
