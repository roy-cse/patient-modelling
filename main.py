import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def remove_outliers(df):
    numerical_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
                  'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def feature_normalization(df):
    scaler = MinMaxScaler()
    numerical_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
                      'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def main():
    data = pd.read_csv('diabetic_data.csv')

    print("Shape of the data:\n", data.shape)
    
    data.drop('encounter_id', axis=1, inplace=True)
    
    missing_values_before = data.isnull().sum()
    print("\nSummary of missing values before replace:\n", missing_values_before)
    
    data.replace('?', np.nan, inplace=True)

    missing_values_after = data.isnull().sum()
    print("\nSummary of missing values after replace:\n", missing_values_after)

    data['readmitted'] = data['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})
    
    print("\nData types of each column:\n", data.dtypes)

    missing_percent = data.isnull().mean() * 100
    
    columns_to_drop = missing_percent[missing_percent > 90].index
    data.drop(columns=columns_to_drop, inplace=True)
    
    columns_to_delete = [
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'payer_code', 'patient_nbr']

    data.drop(columns=columns_to_delete, inplace=True)
    
    data.dropna(axis=0, how='all', inplace=True)
    
    print("\nSummary statistics of numerical columns:\n", data.describe())
    
    # Removing outliers
    data = remove_outliers(data)
    
    # Feature normalization
    data = feature_normalization(data)
    
    print("\nFinal shape of the data:\n", data.shape)

    # data.to_csv('processed_data.csv', index=False)

if __name__ == '__main__':
    main()