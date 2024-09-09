import seaborn as sns
import matplotlib.pyplot as plt

def drop_unnecessary_features(df_train, df_test):
    drop_columns = ["engine", "setting_1", "setting_2", "setting_3", "s1", "s5", "s6", "s9", "s10", "s16", "s18", "s19"]
    df_train.drop(columns=drop_columns, inplace=True)
    df_test.drop(columns=drop_columns, inplace=True)
    df_train.to_csv(
        "/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/processed/df_train_processed.csv",
        index=False)
    df_test.to_csv(
        "/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/processed/df_test_processed.csv",
        index=False)
    return df_train, df_test


def plot_correlation_matrices(df):
    corr_methods = ['pearson']

    for method in corr_methods:
        corr_matrix = df.corr(method=method)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.show()