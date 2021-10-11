from imports import *
import pandas as pd
from primitive_baselines import *

# all the test set here has to be labeled!
models = ["RANDOM", "DET0", "DET1", "DET2", "SEMI1", "SEMI2", "OUR MODEL"]
type_of_model = ["random", "deterministic", "deterministic", "deterministic", "naïve Bayesian", "naïve Bayesian", "contextual"]
if __name__ == "__main__":
    # setting up the data for the primitive baselines
    df_train = pd.read_csv(RAW_TRAIN_DATA_PATH) # without validation: Apples and oranges comparison otherwise
    df_test = pd.read_csv(RAW_TEST_DATA_PATH)
    
    df_train = df_train[["Post", "NP", "sentiment"]]
    df_test = df_test[["Post", "NP", "sentiment"]]

    df_train = df_train.dropna()
    df_test = df_test.dropna()

    df_train['raw'] = df_train["sentiment"].apply(lambda row : MAPPING[row])
    df_test['raw'] = df_test["sentiment"].apply(lambda row : MAPPING[row])

    full_res = []
    for i,m in enumerate(models):
        if m != "OUR MODEL":
            full_res.append([m, type_of_model[i], get_score((df_train["NP"], df_train["raw"]), (df_test["NP"], df_test["raw"]), m, None)])
        else:
            full_res.append([m, type_of_model[i], get_score((None,None), (df_test["NP"], df_test["raw"]), m, df_test["Post"])])

    print()
    print()
    print ("{:<25} | {:<25} | {:<25}".format('Model', 'Type', 'Accuracy'))
    for res in full_res:
        print ("{:<25} | {:<25} | {:<25}".format(res[0], res[1], "{:.3f}".format(res[2])))
    print()
    print()