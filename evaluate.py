from cleaner import *
from simpletransformers.classification import ClassificationModel, ClassificationArgs


def predict():
    model = ClassificationModel("roberta", "./models/roberta_base_1/")
    df_dev = read_json('./original_data/dev.json')
    df_test = read_json('./original_data/eval.json')
    df_predicted_dev = df_dev.copy()[['text', 'reply']]
    df_predicted_test = df_test.copy()[['text', 'reply']]

    df_predicted_dev.columns = ['text_a', 'text_b']
    df_predicted_test.columns = ['text_a', 'text_b']

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


if __name__ == '__main__':
    predict()
