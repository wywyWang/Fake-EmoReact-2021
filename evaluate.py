from cleaner import *
from simpletransformers.classification import ClassificationModel, ClassificationArgs


MODEL_TYPE = "deberta"


def predict_dev(MODEL_PATH, save_path):
    model = ClassificationModel(MODEL_TYPE, MODEL_PATH, use_cuda=True)
    df_dev = read_json('./processed_data/preprocess_my_dev.json')
    # df_dev = read_json('./original_data/dev.json')
    df_predicted_dev = df_dev.copy()[['text', 'reply']]
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
    df_dev[['idx', 'context_idx', 'label']].to_csv(save_path, index=False)


def predict_test(MODEL_PATH, save_path):
    model = ClassificationModel(MODEL_TYPE, MODEL_PATH, use_cuda=True)
    df_test = read_json('./processed_data/preprocess_eval.json')
    # df_test = read_json('./original_data/eval.json')
    df_predicted_test = df_test.copy()[['text', 'reply']]
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
    df_test[['idx', 'context_idx', 'label']].to_csv(save_path, index=False)


if __name__ == '__main__':
    checkpoints_step = [_ for _ in range(2000, 64001, 2000)]
    for checkpoint in checkpoints_step:
        print("==== {} ====".format(checkpoint))
        SAVE_DEV_PATH = "./prediction_lists/dev-{}.csv".format(checkpoint)
        SAVE_EVAL_PATH = "./prediction_lists/eval-{}.csv".format(checkpoint)
        MODEL_PATH = "./models/deberta_base_lm_original_11/checkpoint-{}/".format(checkpoint)
        predict_dev(MODEL_PATH, SAVE_DEV_PATH)
        predict_test(MODEL_PATH, SAVE_EVAL_PATH)