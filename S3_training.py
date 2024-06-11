from accelerate import Accelerator
from transformers import TrainingArguments, Trainer
from config import get_tokenizer
from S1_preprocessing import get_mapped_and_tokenized_dataset_cached
from S2_model_build import get_model
# from torch.utils.data import DataLoader

def perform_training():


    tokenizer = get_tokenizer()
    model = get_model()

    _, tokenized_dataset = get_mapped_and_tokenized_dataset_cached()

    tokenized_dataset.set_format("torch")

    train_batch_size = 4
    eval_batch_size = 4

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Prepare the datasets and the model for the accelerator
    train_dataloader = tokenized_dataset['train']
    eval_dataloader = tokenized_dataset['validation_matched']

    # train_dataloader = DataLoader(
    #     tokenized_dataset["train"], shuffle=True, batch_size=train_batch_size
    # )
    # eval_dataloader = DataLoader(
    #     tokenized_dataset["validation_matched"], shuffle=False, batch_size=eval_batch_size
    # )

    # Initialize the accelerator
    accelerator = Accelerator()
    model, train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(eval_results)

if __name__ == '__main__':
    perform_training()