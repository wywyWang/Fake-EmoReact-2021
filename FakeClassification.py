from FakeModel import *
from FakeDataset import FakeEmoDataset
from cleaner import *
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DebertaTokenizer, RobertaTokenizer
from transformers import DebertaForSequenceClassification, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np

import os
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

MODEL_NAME = 'deberta'
MAX_LENGTH = 113


def compute_metrics(pred):    
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,

        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro
    }


def fine_tuning():
    # df_train = read_json('./processed_data/preprocess_train.json')[['text', 'reply', 'label']].sample(frac=1).reset_index(drop=True)
    # df_dev = read_json('./processed_data/preprocess_my_dev.json')[['text', 'reply', 'label']]

    df_train = read_json('./original_data/train.json')[['text', 'reply', 'label']].sample(frac=1).reset_index(drop=True)
    df_dev = read_json('./processed_data/my_dev.json')[['text', 'reply', 'label']]

    category = {
        'fake': 0,
        'real': 1
    }
    df_train['label'] = df_train['label'].map(category)
    df_dev['label'] = df_dev['label'].map(category)

    if MODEL_NAME == 'deberta':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base', max_length=MAX_LENGTH)
        model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base')
    elif MODEL_NAME == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', max_length=MAX_LENGTH)
        model = RobertaForSequenceClassification.from_pretrained('language_models/roberta_base_1/checkpoint-10000/')
    else:
        raise NotImplementedError

    train_encodings = tokenizer(df_train['text'].values.tolist(), df_train['reply'].values.tolist(), truncation=True, padding=True, return_tensors='pt')
    dev_encodings = tokenizer(df_dev['text'].values.tolist(), df_dev['reply'].values.tolist(), truncation=True, padding=True, return_tensors='pt')
    train_dataset = FakeEmoDataset(train_encodings, df_train['label'])
    dev_dataset = FakeEmoDataset(dev_encodings, df_dev['label'])

    # for name, param in model.named_parameters():
    #     if MODEL_NAME in name: # classifier layer
    #         param.requires_grad = False
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # 1/0

    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-7,

        output_dir='./models/deberta_base_original_6/',
        report_to='tensorboard',
        logging_dir='./models/deberta_base_original_6/',
        logging_strategy='steps',
        seed=42,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        save_strategy='steps'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()


def evaluating():
    df_dev = read_json('./processed_data/preprocess_my_dev.json')[['text', 'reply', 'label']]

    category = {
        'fake': 0,
        'real': 1
    }
    df_dev['label'] = df_dev['label'].map(category)

    if MODEL_NAME == 'deberta':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base', return_token_type_ids=True)
        model = DebertaForSequenceClassification.from_pretrained('./models/deberta_base_2/checkpoint-48000/')
    elif MODEL_NAME == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', return_token_type_ids=True)
        model = RobertaForSequenceClassification.from_pretrained('language_models/roberta_base_1/checkpoint-10000/')
    else:
        raise NotImplementedError

    dev_encodings = tokenizer(df_dev['text'].values.tolist(), df_dev['reply'].values.tolist(), truncation=True, padding=True, return_tensors='pt')
    dev_dataset = FakeEmoDataset(dev_encodings, df_dev['label'])

    training_args = TrainingArguments(
        output_dir='./models/roberta_base_lm_1/',
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
        compute_metrics=compute_metrics
    )

    prediction_results = trainer.evaluate(dev_dataset)
    print(prediction_results)


def predicting():
    # df_test = read_json('./processed_data/preprocess_eval.json')
    df_test = read_json('./original_data/eval.json')

    if MODEL_NAME == 'deberta':
        tokenizer = DebertaTokenizer.from_pretrained('./models/deberta_base_2/checkpoint-48000/', return_token_type_ids=True)
        model = DebertaForSequenceClassification.from_pretrained('./models/deberta_base_2/checkpoint-48000/')
    elif MODEL_NAME == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', return_token_type_ids=True)
        model = RobertaForSequenceClassification.from_pretrained('language_models/roberta_base_1/checkpoint-10000/')
    else:
        raise NotImplementedError

    test_encodings = tokenizer(df_test['text'].values.tolist(), df_test['reply'].values.tolist(), truncation=True, padding=True, return_tensors='pt')
    test_dataset = FakeEmoDataset(test_encodings)

    trainer = Trainer(
            model=model
    )

    predictions = trainer.predict(test_dataset)
    predicted_labels = []
    for prediction in predictions.predictions:
        if np.argmax(prediction) == 0:
            predicted_labels.append('fake')
        else:
            predicted_labels.append('real')
    
    df_test['label'] = predicted_labels
    print(df_test['label'].value_counts())
    df_test[['idx', 'context_idx', 'label']].to_csv('eval.csv', index=False)


if __name__ == '__main__':
    fine_tuning()
    # evaluating()
    # predicting()
