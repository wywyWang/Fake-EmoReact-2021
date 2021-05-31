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
    model_args = LanguageModelingArgs(
        num_train_epochs=3,
        train_batch_size=8,
        eval_batch_size=8,
        max_seq_length=113,
        learning_rate=5e-7,
        evaluate_during_training=True,
        evaluate_during_training_steps=4000,
        tensorboard_dir='./language_models/deberta_base_original_2/',

        output_dir='./language_models/deberta_base_original_2/',
        manual_seed=2021,
        use_multiprocessing=False,
        save_steps=5000,
        n_gpu=1
    )

    model = LanguageModelingModel("deberta", "microsoft/deberta-base", args=model_args)
    model.train_model('LM/training.txt', eval_file='LM/dev.txt')


if __name__ == '__main__':
    train_language_model()