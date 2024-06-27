
from transformers import LlamaTokenizerFast, MistralForSequenceClassification

from tokenizer_for_classification import LlamaTokenizerZeroShotClassifierFast

hf_repo = 'h2oai/h2o-danube2-1.8b-base'

tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(hf_repo)


premise = 'I am'
hypothesis = 'stupid'


tokenized_result = tokenizer(premise, hypothesis, add_special_tokens=True)
print(tokenized_result)

decoded_result = tokenizer.decode(tokenized_result['input_ids'])
print(decoded_result)

