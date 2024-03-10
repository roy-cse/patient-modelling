import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

def process_data(data):
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
    
    # TODO: Drop columns such as payer_code, patient_nbr
    columns_to_delete = [
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone']

    data.drop(columns=columns_to_delete, inplace=True)
    
    data.dropna(axis=0, how='any', inplace=True)
    # TODO: Check if A1CResult plays a major role in predicting the target variable 
    # TODO: Check if removed column values can be replaced with mean or mode
    print("\nSummary statistics of numerical columns:\n", data.describe())
    
    return data

def remove_outliers(df, numerical_cols, threshold=1.5):
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def feature_normalization(df, numerical_cols):
    # List of columns to exclude from normalization
    cols_to_exclude = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id','number_outpatient', 'number_emergency', 'number_inpatient', 'patient_nbr']
    # Exclude specified columns from the list of numerical columns to normalize
    cols_to_normalize = [col for col in numerical_cols if col not in cols_to_exclude]
    print("\nColumns to normalize:\n", cols_to_normalize)    
    # Normalize numerical columns
    # TODO: Check which numerical cols need to be normalized    
    scaler = MinMaxScaler()
    print('11111111111111111111111111111',df.loc[:, ['patient_nbr','number_outpatient', 'number_emergency', 'number_inpatient']].head(100))
    # Normalize only the columns that are not excluded
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    return df

def data_visualisation(data):
    # distribution of unique classes of the target variable
    ax = sns.barplot(x='readmitted', y='readmitted', estimator=lambda x: len(x) / len(data) * 100, 
                     data=data, hue="readmitted", legend=False)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.f%%')

    ax.set_ylabel('Percentage (%)')
    sns.set_theme(rc={"figure.figsize":(10, 7)})
    plt.show()

    # count of number of readmitted cases against age

    # TODO: Check if categorical data can be converted to numerical data
    print("\nCount of number of readmitted cases against age:\n")
    value_counts = data.groupby('age')['readmitted'].value_counts().unstack()

    fig, ax3 = plt.subplots()
    bars = ax3.bar(data["age"].unique(), value_counts[1])
    ax3.bar_label(bars)
    
    plt.xlabel("Age Groups")
    plt.ylabel("Readmitted Cases")
    plt.title("Count of number of readmitted cases against age")
    plt.show()

    # count of target variable against the number of medications
    print("\nCount of target variable against the number of medications:\n")
    ax2 = sns.countplot(x="num_medications", data=data, hue="readmitted", legend=False)
    
    for container in ax2.containers:
        ax2.bar_label(container)

    ax2.get_legend_handles_labels()
    target_unique_classes = data['readmitted'].value_counts().index
    ax2.legend(labels=target_unique_classes, title="readmitted", loc="upper right")
    
    sns.set_theme(rc={"figure.figsize":(10, 7)})
    plt.xticks(rotation=45, ha='right')
    plt.show()

def main():
    data = pd.read_csv('diabetic_data.csv')
    data = process_data(data)
    
    target_var = 'readmitted'
    # Removing outliers
    numerical_cols = data.select_dtypes(include='number').columns
    # Exclude a particular column
    numerical_cols = numerical_cols.drop(target_var)
    data = remove_outliers(data, numerical_cols, threshold=1.5)
    # Feature normalization
    data = feature_normalization(data, numerical_cols)
    
    print("\nFinal shape of the data:\n", data.shape)

    # data.to_csv('processed_data.csv', index=False)

    # Data Visualisation
    data_visualisation(data)

if __name__ == '__main__':
    main()
