import os
import argparse
import pandas as pd
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", type=str, help="path to input data folder")
    parser.add_argument("--train_data", type=str, help="path to output train data folder")
    parser.add_argument("--test_data", type=str, help="path to output test data folder")
    args = parser.parse_args()

    df_train_2011 = pd.read_csv(os.path.join(args.data_files, "Train/gt_2011.csv"))
    df_train_2012 = pd.read_csv(os.path.join(args.data_files, "Train/gt_2012.csv"))
    df_train_2013 = pd.read_csv(os.path.join(args.data_files, "Train/gt_2013.csv"))
    
    df_train_full = pd.concat([df_train_2011, df_train_2012, df_train_2013])

    X_train_full = df_train_full.drop(["CO","NOX"],axis=1)
    y_train_full = df_train_full[["CO"]]

    X_train_full.to_csv(os.path.join(args.train_data,"train_X.csv"), index=False)
    y_train_full.to_csv(os.path.join(args.train_data,"train_Y.csv"), index=False)

    df_test_2014 = pd.read_csv(os.path.join(args.data_files, "Test/gt_2014.csv"))
    df_test_2015 = pd.read_csv(os.path.join(args.data_files, "Test/gt_2015.csv"))
    
    df_test_full = pd.concat([df_test_2014, df_test_2015])

    X_test_full = df_test_full.drop(["CO","NOX"],axis=1)
    y_test_full = df_test_full[["CO"]]

    X_test_full.to_csv(os.path.join(args.test_data,"test_X.csv"), index=False)
    y_test_full.to_csv(os.path.join(args.test_data,"test_Y.csv"), index=False)

if __name__ == "__main__":
    main()
