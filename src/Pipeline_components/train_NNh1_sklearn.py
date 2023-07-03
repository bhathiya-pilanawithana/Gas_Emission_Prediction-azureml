import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, required=False, default=0)
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--scaler_type", type=str, required=False, default="minmax")
    parser.add_argument("--hidden_layer_neurons", type=int, required=False, default=100)
    parser.add_argument("--hidden_layer_activation", type=str, required=False, default="relu")
    parser.add_argument("--learning_rate", type=float, required=False, default=10)
    parser.add_argument("--epochs", type=int, required=False, default=30)
    parser.add_argument("--momentum", type=float, required=False, default=0)
    parser.add_argument("--ouput_model_path", type=str, help="Path for the model")
    args = parser.parse_args()

    X_full = pd.read_csv(os.path.join(args.train_data, "train_X.csv"))
    y_full = pd.read_csv(os.path.join(args.train_data, "train_Y.csv"))

    mlflow.start_run()
    mlflow.sklearn.autolog()

    X_train, X_vali, y_train, y_vali =  train_test_split(X_full,y_full,test_size=args.test_train_ratio,random_state=args.random_state)
 
    model_sklearnNN = MLPRegressor(
        hidden_layer_sizes=args.hidden_layer_neurons, 
        activation=args.hidden_layer_activation, 
        solver="sgd", 
        learning_rate="adaptive", 
        learning_rate_init=args.learning_rate,  
        max_iter=args.epochs,
        momentum=args.momentum,
        random_state=args.random_state
        )

    if (args.scaler_type == "minmax"):
        scaler = MinMaxScaler()
        scaler_model_pipeline = Pipeline([
            ("scalar", scaler),
            ("nn_model", model_sklearnNN)
            ])
    elif (args.scaler_type == "standard"):
        scaler = StandardScaler()
        scaler_model_pipeline = Pipeline([
            ("scalar", scaler),
            ("nn_model", model_sklearnNN)
            ])
    elif (args.scaler_type == "maxabs"):
        scaler = MaxAbsScaler()
        scaler_model_pipeline = Pipeline([
            ("scalar", scaler),
            ("nn_model", model_sklearnNN)
            ])
    else:
        scaler_model_pipeline = model_sklearnNN

    scaler_model_pipeline.fit(X_train, y_train)

    y_predict = scaler_model_pipeline.predict(X_train)
    mlflow.log_metric("Train RMSE",  np.sqrt(mean_squared_error(y_train,y_predict)))
    mlflow.log_metric("Train R2-score", r2_score(y_train,y_predict))

    y_predict = scaler_model_pipeline.predict(X_vali)
    mlflow.log_metric("Validation RMSE",  np.sqrt(mean_squared_error(y_vali,y_predict)))
    mlflow.log_metric("Validation R2-score", r2_score(y_vali,y_predict))

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=scaler_model_pipeline,
        path=args.ouput_model_path,
    )
    mlflow.end_run()

if __name__ == "__main__":
    main()
