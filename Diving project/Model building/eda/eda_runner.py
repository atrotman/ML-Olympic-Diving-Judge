import pandas as pd
from data_overview import data_overview
from univariate_analysis import univariate_analysis
from bivariate_analysis import bivariate_analysis
from correlation_analysis import correlation_analysis

def run_eda(df):
    # Data Overview
    data_overview(df)

    # Univariate Analysis
    numerical_features = ['difficulty', 'rotations', 'twists', 'splash_width', 'splash_height', 'score']
    univariate_analysis(df, numerical_features)

    # Bivariate Analysis
    bivariate_analysis(df)

    # Correlation Analysis
    correlation_analysis(df)

# Usage
if __name__ == "__main__":
    df = pd.read_json('Model building/Preprocessing/preprocessed_data.json')
    run_eda(df)
