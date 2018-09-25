import pandas as pd

def missing_data(dataset):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent_1 = dataset.isnull().sum()/dataset.isnull().count()*100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    return pd.concat([total, percent_2], axis=1, keys=['筆數', '%'])