import xgboost as xgb

def train_xgboost(X_train, y_train, params):
    model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
