import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df, method='pearson'):
    corr_matrix = df.corr(method=method)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f'{method.capitalize()} Correlation Matrix')
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, dataset_name=""):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel(f'Actual RUL ({dataset_name})')
    plt.ylabel(f'Predicted RUL ({dataset_name})')
    plt.title(f'{dataset_name}: Actual vs Predicted')
    plt.show()
