import pandas as pd
import json
from sklearn.model_selection import train_test_split


def read_json(filename):
    with open(filename) as f:
        data = json.load(f)

    return pd.DataFrame(data)