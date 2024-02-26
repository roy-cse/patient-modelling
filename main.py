import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def remove_outliers(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print("Numerical columns:", numerical_cols)
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def feature_normalization(df):
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def main():
    data = pd.read_csv('diabetic_data.csv')

    print("Initial shape of the data:", data.shape)
    
    data.drop('encounter_id', axis=1, inplace=True)
    
    missing_values_before = data.isnull().sum()
    print("Summary of missing values before replace:\n", missing_values_before)
    
    data.replace('?', np.nan, inplace=True)

    missing_values_after = data.isnull().sum()
    print("Summary of missing values after replace:\n", missing_values_after)

    data['readmitted'] = data['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})
    
    print("Data types of each column:\n", data.dtypes)

    missing_percent = data.isnull().mean() * 100
    print("Percentage of missing values:\n", missing_percent)
    
    columns_to_drop = missing_percent[missing_percent > 90].index
    data.drop(columns=columns_to_drop, inplace=True)
    
    columns_to_delete = [
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone']

    data.drop(columns=columns_to_delete, inplace=True)
    
    data.dropna(inplace=True)
    
    print("Summary statistics of numerical columns:\n", data.describe())
    
    # Removing outliers
    data = remove_outliers(data)
    
    # Feature normalization
    data = feature_normalization(data)
    
    print("Final shape of the data:", data.shape)


if __name__ == '__main__':
    main()