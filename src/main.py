import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.data_processing import data_loader, data_cleaning, data_preprocessing
from src.feature_engineering import feature_engineering
from src.modeling import model_training, model_evaluation, model_tuning, model_saving
from src.utils import metrics, visualization


def main(train_path, test_path,rul_path,col_names,dnames,rnames,input_path,output_path):
    #  workflow here

    #load raw data
    data_loader.load_raw_data(dnames,rnames,input_path,output_path)

    # Load Processed data
    df_train, df_test, df_test_RUL = data_loader.load_processed_data(train_path, test_path, rul_path, col_names)

    # Clean and preprocess data
    df_train, df_test = data_cleaning.clean_data(df_train, df_test, df_test_RUL)

    # Feature engineering
    feature_engineering.plot_correlation_matrices(df_train) # correlation

    df_train, df_test = feature_engineering.drop_unnecessary_features(df_train, df_test)

    # Train model
    X_train, y_train = df_train.drop('RUL', axis=1), df_train['RUL']
    X_test, y_test = df_test.drop(columns=['RUL']), df_test[['RUL']]
    scaler = MinMaxScaler()

    # Fit the scaler on the training data_processing and transform both training and test data_processing
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Transform target variable
    y_train_normalized = np.log(y_train + 1)
    y_test_normalized = np.log(y_test + 1)

    X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train_normalized, y_train_normalized,
                                                                    test_size=0.2, random_state=42)

    params = model_tuning.tune_model(X_train, y_train)
    model = model_training.train_xgboost(X_train, y_train, params)

    # Save the model
    model_saving.save_model(model, '/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/models/xgboost.json')

    # Evaluation
    y_pred_train = model.predict(X_train)
    r2, mse = model_evaluation.evaluate_model(y_train, y_pred_train)

    print(f"Training R2: {r2}, MSE: {mse}")


if __name__ == '__main__':
    train_path = '/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/interim/train.txt'
    test_path = '/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/interim/test.txt'
    rul_path = '/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/interim/RUL.txt'
    index_names = ['engine', 'cycle']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
                    "s16", "s17", "s18", "s19", "s20", "s21"]
    col_names = index_names + setting_names + sensor_names
    dnames = ["train_FD001", "test_FD001", "RUL_FD001"]
    rnames = ["train", "test", "RUL"]
    input_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/raw/"
    output_path = '/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/interim/'
    main(train_path, test_path,rul_path,col_names,dnames,rnames,input_path,output_path)
