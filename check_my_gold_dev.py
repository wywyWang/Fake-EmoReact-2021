from sklearn.metrics import f1_score
import pandas as pd


def evaluate_dev():
    df_predicted = pd.read_csv('./prediction_lists_preprocess_input-deberta_base_lm_original_11/dev1-14000.csv')
    df_my_gold = pd.read_csv('./my_gold_dev.csv')

    print(f1_score(df_my_gold['label'], df_predicted['label'], average='macro'))


if __name__ == '__main__':
    evaluate_dev()