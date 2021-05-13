from DeBERTa import deberta
from FakeModel import *
from FakeDataset import FakeEmoDataset
import torch
from cleaner import *
from transformers import RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from datasets import Dataset


def tokenize_function(examples):
    return tokenizer(examples['text'], examples['reply'], truncation=True, padding=True)


def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
    # print(examples)
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def fine_tuning():
    df_train = read_json('./processed_data/preprocess_train.json')[['text', 'reply', 'label']].sample(frac=1).reset_index(drop=True)
    df_dev = read_json('./processed_data/preprocess_my_dev.json')[['text', 'reply', 'label']]

    category = {
        'fake': 0,
        'real': 1
    }
    df_train['label'] = df_train['label'].map(category)
    df_dev['label'] = df_dev['label'].map(category)

    global tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', return_token_type_ids=True)
    
    datasets_train = Dataset.from_pandas(df_train)
    tokenized_datasets_train = datasets_train.map(tokenize_function, batched=True, num_proc=8, remove_columns=['text', 'reply', 'label'])
    datasets_dev = Dataset.from_pandas(df_dev)
    tokenized_datasets_dev = datasets_dev.map(tokenize_function, batched=True, num_proc=8, remove_columns=['text', 'reply', 'label'])


    lm_datasets_train = tokenized_datasets_train.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=8
    )
    lm_datasets_dev = tokenized_datasets_dev.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=8
    )


    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-6,

        output_dir='./language_models/roberta_base_1/',
        logging_dir='./language_models/roberta_base_1/',
        logging_strategy='steps',
        report_to='tensorboard',
        seed=42,
        evaluation_strategy='steps',
        eval_steps=2000,
        save_steps=2000,
        save_strategy='steps'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets_train,
        eval_dataset=lm_datasets_dev,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == '__main__':
    fine_tuning()