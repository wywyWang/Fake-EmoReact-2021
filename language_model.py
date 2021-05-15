from cleaner import *
from simpletransformers.language_modeling import (
    LanguageModelingModel, LanguageModelingArgs
)
import torch
import sklearn
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def train_language_model():
    df_train = read_json('./original_data/train.json')[['text', 'reply', 'label']].sample(frac=1).reset_index(drop=True)
    df_dev = read_json('./processed_data/my_dev.json')[['text', 'reply', 'label']]

    model_args = LanguageModelingArgs(
        num_train_epochs=3,
        train_batch_size=8,
        eval_batch_size=8,
        max_seq_length=113,
        learning_rate=5e-7,
        tensorboard_dir='./language_models/deberta_base_2/',

        output_dir='./language_models/deberta_base_2/',
        manual_seed=42,
        use_multiprocessing=False,
        save_steps=2000,
        n_gpu=1
    )

    model = LanguageModelingModel("deberta", "microsoft/deberta-base", args=model_args)
    model.train_model('LM/training.txt')


if __name__ == '__main__':
    train_language_model()