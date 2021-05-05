from cleaner import *

# from transformers import BartTokenizer, BartForSequenceClassification
# from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
import sklearn
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


if __name__ == '__main__':
    df_train = read_json('./original_data/train.json')[['text', 'reply', 'label']]
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
        num_train_epochs=3,
        train_batch_size=16,
        eval_batch_size=16,
        max_seq_length=113,
        evaluate_during_training=True,
        evaluate_during_training_steps=5000,
        learning_rate=4e-5,

        output_dir='./models/roberta_base_5/',
        manual_seed=42,
        use_multiprocessing=False,
        save_steps=5000,
        n_gpu=1
    )
    model = ClassificationModel("roberta", "roberta-base", args=model_args)

    model.train_model(df_train, eval_df=df_dev, f1=sklearn.metrics.f1_score)
