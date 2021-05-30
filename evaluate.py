from cleaner import *
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import os
import sys


MODEL_TYPE = "deberta"


def predict_dev_round_1(MODEL_PATH, save_path):
    model = ClassificationModel(MODEL_TYPE, MODEL_PATH, use_cuda=True)
    # df_dev = read_json('./processed_data/my_dev.json')
    df_dev = read_json('./processed_data/preprocess_my_dev.json')
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
    log.write(str(df_dev['label'].value_counts()))
    log.write('\n')
    df_dev[['idx', 'context_idx', 'label']].to_csv(save_path, index=False)


def predict_test_round_1(MODEL_PATH, save_path):
    model = ClassificationModel(MODEL_TYPE, MODEL_PATH, use_cuda=True)
    # may be useful if using preprocessing eval
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
    log.write(str(df_test['label'].value_counts()))
    log.write('\n')
    df_test[['idx', 'context_idx', 'label']].to_csv(save_path, index=False)


def predict_dev_round_2(MODEL_PATH, save_path):
    model = ClassificationModel(MODEL_TYPE, MODEL_PATH, use_cuda=True)
    df_dev = read_json('./processed_data/preprocess_new_dev.json')
    # df_dev = read_json('./original_data/new_dev.json')
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
    log.write(str(df_dev['label'].value_counts()))
    log.write('\n')
    df_dev[['idx', 'context_idx', 'label']].to_csv(save_path, index=False)


def predict_test_round_2(MODEL_PATH, save_path):
    model = ClassificationModel(MODEL_TYPE, MODEL_PATH, use_cuda=True)
    df_test = read_json('./processed_data/preprocess_new_eval.json')
    # df_test = read_json('./original_data/new_eval.json')
    df_predicted_test = df_test.copy()[['text', 'reply']].sample(5)
    df_predicted_test.columns = ['text_a', 'text_b']

    predictions, raw_outputs = model.predict(df_predicted_test.values.tolist())

    labels = []
    for prediction in predictions:
        if prediction == 0:
            labels.append('fake')
        else:
            labels.append('real')

    df_test['label'] = labels
    log.write(str(df_test['label'].value_counts()))
    log.write('\n')
    df_test[['idx', 'context_idx', 'label']].to_csv(save_path, index=False)


if __name__ == '__main__':
    checkpoints_step = [_ for _ in range(2000, 60001, 2000)]
    MODEL = sys.argv[1]
    directory = './final_predictions-{}/'.format(MODEL)
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    global log
    log_path = "{}{}.log".format(directory, MODEL)
    log = open(log_path, 'a')
    
    for checkpoint in checkpoints_step:
        log.write("\n==== {} ====\n".format(checkpoint))
        SAVE_DEV_1_PATH = "{}dev1-{}.csv".format(directory, checkpoint)
        SAVE_EVAL_1_PATH = "{}eval1-{}.csv".format(directory, checkpoint)
        SAVE_DEV_2_PATH = "{}dev2-{}.csv".format(directory, checkpoint)
        SAVE_EVAL_2_PATH = "{}eval2-{}.csv".format(directory, checkpoint)
        MODEL_PATH = "./models_round_1/{}/checkpoint-{}/".format(MODEL, checkpoint)
        
        # predict_dev_round_1(MODEL_PATH, SAVE_DEV_1_PATH)
        # predict_test_round_1(MODEL_PATH, SAVE_EVAL_1_PATH)
        # predict_dev_round_2(MODEL_PATH, SAVE_DEV_2_PATH)
        predict_test_round_2(MODEL_PATH, SAVE_EVAL_2_PATH)
