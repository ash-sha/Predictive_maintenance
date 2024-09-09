import optuna
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def objective(trial, X_train, y_train):
    params = {
        'objective': 'reg:squarederror',
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0)
    }

    model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    return cv_scores.mean()


def tune_model(X_train, y_train, n_trials=60):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    return study.best_params
