def data_overview(df):
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())