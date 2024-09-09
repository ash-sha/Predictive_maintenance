import xgboost as xgb

def save_model(model, path):
    model.save_model(path)

def load_model(path):
    model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    model.load_model(path)
    return model
