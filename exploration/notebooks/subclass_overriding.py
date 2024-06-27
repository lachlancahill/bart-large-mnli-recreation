from transformers import LlamaTokenizerFast


class UpdatedTokenizerFast(LlamaTokenizerFast):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            raise NotImplementedError(
                'This method is designed for zero shot classification, so requires a premise and hypothesis to be passed. No token_ids_1 was passed.')

        output = [self.bos_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.eos_token_id]

        return output


hf_repo = 'h2oai/h2o-danube2-1.8b-base'

premise, hypothesis = 'Thanks', 'again'

tokenizer = UpdatedTokenizerFast.from_pretrained(hf_repo)

example_tokens = tokenizer(premise, hypothesis, add_special_tokens=True)

print(example_tokens)
# {'input_ids': [8868, 1076], 'attention_mask': [1, 1]}

print(tokenizer.decode(example_tokens['input_ids']))
# Thanks again
