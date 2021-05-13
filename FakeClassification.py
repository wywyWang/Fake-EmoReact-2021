from DeBERTa import deberta
from FakeModel import *
from FakeDataset import FakeEmoDataset
from cleaner import *
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DebertaTokenizer
from transformers import DebertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def compute_metrics(pred):    
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall
    }


def fine_tuning():
    df_train = read_json('./processed_data/preprocess_train.json')[['text', 'reply', 'label']].sample(frac=1).reset_index(drop=True)
    df_dev = read_json('./processed_data/preprocess_my_dev.json')[['text', 'reply', 'label']]

    category = {
        'fake': 0,
        'real': 1
    }
    df_train['label'] = df_train['label'].map(category)
    df_dev['label'] = df_dev['label'].map(category)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base', return_token_type_ids=True)

    train_encodings = tokenizer(df_train['text'].values.tolist(), df_train['reply'].values.tolist(), truncation=True, padding=True, return_tensors='pt')
    dev_encodings = tokenizer(df_dev['text'].values.tolist(), df_dev['reply'].values.tolist(), truncation=True, padding=True, return_tensors='pt')

    train_dataset = FakeEmoDataset(train_encodings, df_train['label'])
    dev_dataset = FakeEmoDataset(dev_encodings, df_dev['label'])

    model = DebertaForSequenceClassification.from_pretrained('language_models/deberta_base_1/checkpoint-70000/')

    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-7,
        dataloader_num_workers=4,

        output_dir='./models/deberta_base_lm_4/',
        report_to='tensorboard',
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


def predicting():
    df_test = read_json('./processed_data/preprocess_eval.json')

    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base', return_token_type_ids=True)

    test_encodings = tokenizer(df_test['text'].values.tolist(), df_test['reply'].values.tolist(), truncation=True, padding=True, return_tensors='pt')
    test_dataset = FakeEmoDataset(test_encodings)

    model = DebertaForSequenceClassification.from_pretrained('models/deberta_base_lm_4/checkpoint-70000/')
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
    # fine_tuning()
    predicting()