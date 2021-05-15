from cleaner import *

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
    df_train.columns = ['text_a', 'text_b', 'labels']
    df_dev.columns = ['text_a', 'text_b', 'labels']

    model_args = ClassificationArgs(
        num_train_epochs=3,
        train_batch_size=8,
        eval_batch_size=8,
        max_seq_length=113,
        evaluate_during_training=True,
        evaluate_during_training_steps=2000,
        learning_rate=5e-7,
        tensorboard_dir='./models/deberta_base_original_6/',

        output_dir='./models/deberta_base_original_6/',
        manual_seed=42,
        use_multiprocessing=False,
        save_steps=2000,
        n_gpu=1
    )
    model = ClassificationModel("deberta", "microsoft/deberta-base", args=model_args)

    model.train_model(df_train, eval_df=df_dev, f1=sklearn.metrics.f1_score, f1_macro=lambda truth, predictions: sklearn.metrics.f1_score(truth, predictions, average='macro'), f1_micro=lambda truth, predictions: sklearn.metrics.f1_score(truth, predictions, average='micro'))
