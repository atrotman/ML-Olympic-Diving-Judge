import matplotlib.pyplot as plt
import seaborn as sns

def bivariate_analysis(df):
    # Scatter plot for Score vs. Difficulty
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='difficulty', y='score')
    plt.title('Score vs. Difficulty')
    plt.show()

    # Score by Dive Type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='dive_type', y='score')
    plt.xticks(rotation=45)
    plt.title('Score by Dive Type')
    plt.show()
