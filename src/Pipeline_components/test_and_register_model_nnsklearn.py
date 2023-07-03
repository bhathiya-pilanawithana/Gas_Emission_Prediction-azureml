import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.keras
import mlflow.sklearn

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_folder", type=str, help="Path of the test data folder")
    parser.add_argument("--input_model_path", type=str, help="path of the input model")
    parser.add_argument("--registered_model_name", type=str, help="Name for the registered best model")
    args = parser.parse_args()

    X_test = pd.read_csv(os.path.join(args.test_data_folder, "test_X.csv"))
    y_test = pd.read_csv(os.path.join(args.test_data_folder, "test_Y.csv"))

    mlflow.start_run()
    
    loaded_best_model = mlflow.sklearn.load_model(args.input_model_path)

    print("Registering the best sweeped model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=loaded_best_model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    y_predict = loaded_best_model.predict(X_test)
    mlflow.log_metric("Test RMSE",  np.sqrt(mean_squared_error(y_test,y_predict)))
    mlflow.log_metric("Test R2-score", r2_score(y_test,y_predict))

    mlflow.end_run()


if __name__ == "__main__":
    main()    
