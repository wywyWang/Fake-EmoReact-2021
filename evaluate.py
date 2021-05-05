from cleaner import *
from simpletransformers.classification import ClassificationModel, ClassificationArgs


def predict_dev():
    model = ClassificationModel("roberta", "./models/roberta_base_5/")
    df_dev = read_json('./original_data/dev.json')
    df_predicted_dev = df_dev.copy()[['text', 'reply']]

    # df_predicted_dev['text'] = df_predicted_dev.text.apply(normalize_Tweet)
    # df_predicted_dev['reply'] = df_predicted_dev.reply.apply(normalize_Tweet)
    df_predicted_dev.columns = ['text_a', 'text_b']

    predictions, raw_outputs = model.predict(df_predicted_dev.values.tolist())

    labels = []
    for prediction in predictions:
        if prediction == 0:
            labels.append('fake')
        else:
            labels.append('real')

    df_dev['label'] = labels
    print(df_dev['label'].value_counts())
    df_dev[['idx', 'context_idx', 'label']].to_csv('dev.csv', index=False)


def predict_test():
    model = ClassificationModel("roberta", "./models/roberta_base_5/")
    df_test = read_json('./original_data/eval.json')
    df_predicted_test = df_test.copy()[['text', 'reply']]

    # df_predicted_test['text'] = df_predicted_test.text.apply(normalize_Tweet)
    # df_predicted_test['reply'] = df_predicted_test.reply.apply(normalize_Tweet)
    df_predicted_test.columns = ['text_a', 'text_b']

    predictions, raw_outputs = model.predict(df_predicted_test.values.tolist())

    labels = []
    for prediction in predictions:
        if prediction == 0:
            labels.append('fake')
        else:
            labels.append('real')

    df_test['label'] = labels
    print(df_test['label'].value_counts())
    df_test[['idx', 'context_idx', 'label']].to_csv('eval.csv', index=False)


if __name__ == '__main__':
    # predict_dev()
    predict_test()