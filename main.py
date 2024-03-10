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
    scaler = MinMaxScaler()
    # Normalize numerical columns
    # TODO: Check which numerical cols need to be normalized
    categorical_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    
    # Save categorical columns data
    categorical_data = df[categorical_cols]
    
    # Drop categorical columns
    df = df.drop(columns=categorical_cols)
    
    # Normalize numerical columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    for col in categorical_cols:
        df[col] = categorical_data[col]
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


    # Creating a new DataFrame with only the specified numerical columns
    num_df = data.select_dtypes(include='number')
    print(num_df.dtypes)

    num_df.drop(['admission_type_id','discharge_disposition_id','admission_source_id','number_outpatient','number_emergency','number_inpatient'], axis=1, inplace=True)


    plot_correlation_matrix(num_df)
    plot_scatter_matrix(num_df)

    plot_avg_lab_procedures_by_race(data)

    print(num_df.dtypes)
    print(data.dtypes)


def plot_scatter_matrix(num_df):

    pd.plotting.scatter_matrix(num_df, alpha=0.2, figsize=(20, 20), diagonal='kde')
    plt.suptitle('Scatter Matrix for Selected Numerical Features')
    plt.show()

def plot_correlation_matrix(num_df):

    # Calculate the correlation matrix
    corr_matrix= num_df.corr()

    # Drop the 'NaN' correlations
    plt.figure(figsize=(12, 12))

    # Visualisation the correlation matrix
    sns.heatmap(corr_matrix,annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xticks(rotation=45, ha='right')

    # Add labels for axes
    plt.xlabel('Features')
    plt.ylabel('Features')

    #TODO add labels to  the xticks
    plt.title("Correlation matrix")
    plt.xlabel("Numerical Features")
    plt.ylabel("Numerical Features")
    plt.show()

    # Threshold for high correlation (can be adjusted)
    # TODO Evaluate hıghly corr
    threshold = 0.2
    # Declaration of pairs from matrix
    highly_correlated_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)
    # Removing self-correlation and correlations below than the threshold
    highly_correlated_pairs = highly_correlated_pairs[(abs(highly_correlated_pairs) > threshold) & (highly_correlated_pairs < 1)]

    # Listing highly correlated pairs
    print("Highly correlated pairs:\n", highly_correlated_pairs.to_string())



def plot_avg_lab_procedures_by_race(data):
    # Bar Chart of Average Number of Lab Procedures by Race
    average_lab_procedures_by_race = data.groupby('race')['num_lab_procedures'].mean().reset_index()
    sns.barplot(data=average_lab_procedures_by_race, x="race", y='num_lab_procedures')
    plt.figure(figsize=(12, 12))
    plt.title('Average Number of Lab Procedures by Race')
    plt.xticks(rotation=45)
    plt.show()
def plot_age_frequency(data):
    # Bar chart of age group frequency
    age_group_counts = data['age'].value_counts().reset_index()
    age_group_counts.columns = ['age', 'count']
    age_group_counts.sort_values('age', inplace=True)
    sns.barplot(data=age_group_counts, x='age', y='count')
    plt.title('Frequency of Each Age Group')
    plt.xticks(rotation=45)
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
