
from transformers import BartModel

import torch.nn as nn

def get_model():

    # Load pre-trained BART model
    model = BartModel.from_pretrained('facebook/bart-large')

    class BartForSequenceClassification(nn.Module):
        def __init__(self, base_model, num_labels):
            super().__init__()
            self.base_model = base_model
            self.classifier = nn.Linear(base_model.config.d_model, num_labels)

        def forward(self, input_ids, attention_mask=None):
            outputs = self.base_model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            cls_output = last_hidden_state[:, 0]  # Take the output of the [CLS] token
            logits = self.classifier(cls_output)
            return logits

    model = BartForSequenceClassification(model, num_labels=3)

    return model

if __name__ == '__main__':
    model = get_model()
    print(model)
