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
    df_dev = read_json('./original_data/dev.json')[['text', 'reply']]
    df_test = read_json('./original_data/eval.json')[['text', 'reply']]

    codes_type, uniques_type = pd.factorize(df_train['label'])
    df_train['label'] = codes_type
    df_train.columns = ['text_a', 'text_b', 'labels']
    train_data, val_data = train_test_split(df_train, test_size=.2)

    model_args = ClassificationArgs(
        num_train_epochs=4,
        train_batch_size=16,
        eval_batch_size=16,
        evaluate_during_training=True,
        evaluate_during_training_steps=1000,
        learning_rate=4e-5,
        output_dir='./models/roberta_base_1/',
        manual_seed=42
    )
    model = ClassificationModel("roberta", "roberta-base", args=model_args)

    model.train_model(train_data, eval_df=val_data, f1=sklearn.metrics.f1_score)
