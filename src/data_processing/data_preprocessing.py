from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized

def log_transform_target(y_train, y_test):
    y_train_normalized = np.log(y_train + 1)
    y_test_normalized = np.log(y_test + 1)
    return y_train_normalized, y_test_normalized
