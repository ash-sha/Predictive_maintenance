import pandas as pd

def clean_data(df_train, df_test, df_test_RUL):
    # Add your cleaning operations here
    df_train_RUL = df_train.groupby(['engine']).agg({'cycle':'max'})
    df_train_RUL.rename(columns={'cycle':'life'}, inplace=True)
    df_train = df_train.merge(df_train_RUL, on='engine', how='left')
    df_train['RUL'] = df_train['life'] - df_train['cycle']
    df_train.drop(['life'], axis=1, inplace=True)
    df_train.to_csv(
        "/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/processed/df_train.csv",
        index=False)

    df_test_RUL.columns = ['RUL']
    df_test_RUL['engine'] = range(1, len(df_test_RUL) + 1)
    df_test = df_test.merge(df_test_RUL, on='engine', how='left')
    df_test.to_csv(
        "/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/processed/df_test.csv",
        index=False)


    return df_train, df_test
