import matplotlib.pyplot as plt
import seaborn as sns

def univariate_analysis(df, numerical_features):
    sns.set(style="whitegrid")
    for feature in numerical_features:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[feature], kde=True, bins=20)
        plt.title(f'Distribution of {feature}')
        plt.show()
