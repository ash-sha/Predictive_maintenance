import pytest
from src.modeling.model_training import train_xgboost

def test_train_model():
    X_train = [[1, 2], [3, 4], [5, 6]]
    y_train = [1, 2, 3]
    params = {'objective': 'reg:squarederror', 'booster': 'gbtree', 'n_estimators': 50, 'max_depth': 3}
    model = train_xgboost(X_train, y_train, params)
    assert model is not None, "Model not trained properly"
