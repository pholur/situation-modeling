from imports import *
import pandas as pd
from primitive_baselines import *
from data import *
from tqdm import tqdm

#metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# all the test set here has to be labeled!
models = ["RANDOM", "DET_0", "DET_1", "DET_2", "SEMI1_lemm", "SEMI2_exact", "CBOW-1", "CBOW-2", "CBOW-5", "OUR MODEL"]
#models = ["OUR MODEL"]
#type_of_model = ["contextual"]
type_of_model = ["random", "deterministic", "deterministic", "deterministic", "naïve Bayesian", "naïve Bayesian", "C: GLoVE300 + XGB", "C: GLoVE300 + XGB", "C: GLoVE300 + XGB", "C: BERT I/O Model"]
if __name__ == "__main__":
    # setting up the data for the primitive baselines
    df_train = pd.read_csv(RAW_TRAIN_DATA_PATH) # without validation: Apples and oranges comparison otherwise
    df_test = pd.read_csv(RAW_TEST_DATA_PATH)
    df_validation = pd.read_csv(RAW_VAL_DATA_PATH)

    if PAIR_TEST_WITH_LEAVE_OUT:
        # # leave out NP phrase matches from test df and check
        list_of_seen_nps = list(df_train["NP"])
        list_of_seen_nps = [nip.lower() for nip in list_of_seen_nps]
        list_of_seen_nps = [clean_text(nip) for nip in list_of_seen_nps]
        list_of_seen_nps = [clean2(nip) for nip in list_of_seen_nps]
        set_of_nps = set(list_of_seen_nps)

        # # filter test set such that the noun phrase does not appear in set_of_nps
        df_test["processed_NP"] = df_test["NP"].apply(lambda x: clean_text(x))
        df_test["processed_NP"] = df_test["processed_NP"].apply(lambda x: x.lower())
        df_test["processed_NP"] = df_test["processed_NP"].apply(lambda x: clean2(x))
        df_test = df_test[~df_test["processed_NP"].isin(set_of_nps)]
        print("Number of test examples after filtering: ", len(df_test))

    df_train = df_train[["Post", "NP", "sentiment"]]
    df_train.append(df_validation[["Post", "NP", "sentiment"]])
    df_test = df_test[["Post", "NP", "sentiment"]]

    df_train = df_train.dropna()
    df_test = df_test.dropna()

    df_train['raw'] = df_train["sentiment"].apply(lambda row : MAPPING[row])
    df_test['raw'] = df_test["sentiment"].apply(lambda row : MAPPING[row])

    full_res = []
    for i,m in tqdm(enumerate(models)):

        if m not in ["OUR MODEL","CBOW-1","CBOW-2","CBOW-5"]:
            y_pred = get_score((df_train["NP"], df_train["raw"]), (df_test["NP"], df_test["raw"]), m, None, None)
        else:
            y_pred = get_score((df_train["NP"], df_train["raw"]), (df_test["NP"], df_test["raw"]), m, df_test["Post"], df_train["Post"])

        # compute metrics
        acc = accuracy_score(df_test["raw"], y_pred)
        f1 = f1_score(df_test["raw"], y_pred, average="macro")
        prec = precision_score(df_test["raw"], y_pred, average="macro")
        rec = recall_score(df_test["raw"], y_pred, average="macro")
        f1_weighted = f1_score(df_test["raw"], y_pred, average="weighted")
        scores = [acc, prec, rec, f1, f1_weighted]
        full_res.append([m, type_of_model[i], *scores])

    print()
    print()
    print ("{:<15} | {:<35} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format('Model', 'Type', 'Acc', 'P', 'R', 'F1', 'F1 (weighted)'))
    for res in full_res:
        print ("{:<15} | {:<35} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(res[0], res[1], "{:.3f}".format(res[2]), "{:.3f}".format(res[3]), "{:.3f}".format(res[4]), "{:.3f}".format(res[5]), "{:.3f}".format(res[6])))
    print()
    print()