from cleaner import *

# from transformers import BartTokenizer, BartForSequenceClassification
# from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
import sklearn
import logging

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


if __name__ == '__main__':
    df_train = read_json('./original_data/train.json')[['text', 'reply', 'label']].sample(frac=1).reset_index(drop=True)
    df_dev = read_json('./processed_data/my_dev.json')[['text', 'reply', 'label']]

    category = {
        'fake': 0,
        'real': 1
    }

    # # normalize text
    # df_train['text'] = df_train.text.apply(normalize_Tweet)
    # df_train['reply'] = df_train.reply.apply(normalize_Tweet)
    # df_dev['text'] = df_dev.text.apply(normalize_Tweet)
    # df_dev['reply'] = df_dev.reply.apply(normalize_Tweet)

    df_train['label'] = df_train['label'].map(category)
    df_dev['label'] = df_dev['label'].map(category)
    print(df_train['label'].head(), df_dev['label'].head())
    df_train.columns = ['text_a', 'text_b', 'labels']
    df_dev.columns = ['text_a', 'text_b', 'labels']

    model_args = ClassificationArgs(
        num_train_epochs=4,
        train_batch_size=16,
        eval_batch_size=16,
        max_seq_length=113,
        evaluate_during_training=True,
        evaluate_during_training_steps=5000,
        learning_rate=5e-6,

        output_dir='./models/deberta_base_1/',
        manual_seed=42,
        use_multiprocessing=False,
        save_steps=5000,
        n_gpu=1
    )
    model = ClassificationModel("deberta", "microsoft/deberta-base", args=model_args)

    model.train_model(df_train, eval_df=df_dev, f1=sklearn.metrics.f1_score)
