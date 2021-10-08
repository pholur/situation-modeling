from imports import *
import pandas as pd
from baselines import *

# all the test set here has to be labeled!
models = ["RANDOM", "SEMI1", "SEMI2", "OUR MODEL"]
if __name__ == "__main__":
    # setting up the data for the primitive baselines
    df_train = pd.read_csv(RAW_TRAIN_DATA_PATH) # without validation: Apples and oranges comparison otherwise
    df_test = pd.read_csv(RAW_TEST_DATA_PATH)
    df_train = df_train(columns=["NP", "sentiment"])
    df_test = df_test(columns=["NP", "sentiment"])

    df_train['raw'] = df_train.apply(lambda row : MAPPING[row['sentiment']])
    df_test['raw'] = df_test.apply(lambda row : MAPPING[row['sentiment']])

    for m in models:
        print(m, get_score((df_train["NP"], df_train["sentiment"]), (df_test["NP"], df_test["sentiment"]), m))
    # setting up the data for the DL model