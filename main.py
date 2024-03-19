import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


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
    
    # Dropping cols such as payer_code and medical_specialty since they don't play a major role in predicting the target variable
    columns_to_delete = [
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'payer_code', 'medical_specialty']

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
    # Exclude specified columns from the list of numerical columns to normalize
    scaler = MinMaxScaler()
    # Normalize only the columns that are not excluded
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def data_visualisation(data, categorical_int_cols):
    # distribution of unique classes of the target variable
    ax = sns.barplot(x='readmitted', y='readmitted', estimator=lambda x: len(x) / len(data) * 100, 
                     data=data, hue="readmitted", legend=False)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.f%%')

    ax.set_ylabel('Percentage (%)')
    sns.set_theme(rc={"figure.figsize":(10, 7)})
    plt.show()

    # count of number of readmitted cases against age
    value_counts = data.sort_values('age').groupby('age')['readmitted'].value_counts().unstack()    
    fig, ax3 = plt.subplots()

    bars = ax3.bar(value_counts.index, value_counts[1])
    ax3.bar_label(bars)
    
    plt.xlabel("Age Groups")
    plt.ylabel("Readmitted Cases")
    plt.title("Count of number of readmitted cases against age")
    plt.show()

    # count of target variable against the number of medications
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
    num_df.drop(categorical_int_cols, axis=1, inplace=True)

    plot_correlation_matrix(num_df)
    plot_scatter_matrix(num_df)


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

def improved_model(data):
    data.replace('?', np.nan, inplace=True)
    columns_to_delete=['encounter_id','patient_nbr','admission_type_id','discharge_disposition_id','admission_source_id']
    num_cols_for_filling=data.select_dtypes(include='number').drop(columns=columns_to_delete).columns.values
    print('cols',num_cols_for_filling)

    # FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    # The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    # For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.

    for column in num_cols_for_filling:
        data[column].fillna(data[column].mean(), inplace=True)

    new_df_for_clusters=data[['num_lab_procedures','num_procedures']]

    print("sss",new_df_for_clusters)
    # features, true_labels = make_blobs(n_samples=[100, 100, 200], centers=None, cluster_std=[0.7, 0.7, 0.7],
    #                                    random_state=0)
    plt.scatter(data=new_df_for_clusters,x='num_lab_procedures',y='num_procedures')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()

    # UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=2.
    new_df_for_clusters=new_df_for_clusters.values
    kmeans = KMeans(n_clusters=2,init='random',max_iter=100,random_state=10,n_init='auto')
    kmeans.fit(new_df_for_clusters)
    plt.scatter(new_df_for_clusters[kmeans.labels_ == 0, 0], new_df_for_clusters[kmeans.labels_ == 0, 1])
    plt.scatter(new_df_for_clusters[kmeans.labels_ == 1, 0], new_df_for_clusters[kmeans.labels_ == 1, 1])

    plt.scatter(kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[0][1],c='red',marker='x')
    plt.scatter(kmeans.cluster_centers_[1][0],kmeans.cluster_centers_[1][1],c='blue',marker='x')

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()

    print("son",data.isnull().sum())
    print("print",data['number_inpatient'].to_string())

def main():
    data = pd.read_csv('diabetic_data.csv')
    improved_model(data)
    data = process_data(data)
    print('The shape of the data after processing:', data.shape)
    target_var = ['readmitted']
    print('The numerical cols are:', data.select_dtypes(include='number').columns.values)
   
    # Not removing outliers from these columns since they are categorical types

    categorical_int_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
   
   # Not removing outliers from these columns since values are inside 3 standard deviations
    non_outlier_cols = ['number_outpatient', 'number_emergency', 'number_inpatient', 'time_in_hospital', 'num_procedures', 'patient_nbr']

    final_non_outlier_cols = target_var + categorical_int_cols + non_outlier_cols

    print("\nColumns that are not outliers:\n", final_non_outlier_cols)
    
    # Removing outliers
    numerical_cols = data.select_dtypes(include='number').columns
    # Exclude the non-outlier columns
    numerical_cols = numerical_cols.drop(final_non_outlier_cols)

    print('\n Dropping outliers from: ', numerical_cols.values)
    data = remove_outliers(data, numerical_cols, threshold=1.5)

    # We are not performing normalisation on any of the columns since the range of the values for every feature has insignificant difference
    # feature_normalization(data, numerical_cols)
    print("\nFinal shape of the data:\n", data.shape)

    # data.to_csv('processed_data.csv', index=False)

    # Data Visualisation
    # data_visualisation(data, categorical_int_cols)

if __name__ == '__main__':
    main()
