import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.utils import resample

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
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'payer_code', 'medical_specialty', 'patient_nbr']

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
        
    gender_plot = sns.countplot(x = 'gender', data = data, hue = 'readmitted')
    gender_plot.figure.set_size_inches(12, 12)
    gender_plot.legend(title = 'Readmitted', labels = ['No', 'Yes'])
    gender_plot.axes.set_title('Readmission based on Gender')
    plt.show()


    fig, ax = plt.subplots(figsize=(10,15), ncols=1, nrows=3)  # Adjusted for 1 column and 3 rows
    sns.countplot(x="readmitted", data=data, ax=ax[0])  # Plot 1 in the first row
    sns.countplot(x="race", data=data, ax=ax[1])        # Plot 2 in the second row
    sns.countplot(x="gender", data=data, ax=ax[2])      # Plot 3 in the third row
    plt.tight_layout()  # Adjust the layout to make sure there's no overlap
    plt.show()

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


def evaluate_model_performance(data):
    # Convert categorical variables to dummy variables
    cat_cols = ['race', 'gender', 'age', 'admission_type_id' , 'discharge_disposition_id', 'admission_source_id', 'diag_1',
 'diag_2', 'diag_3', 'A1Cresult', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed']
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    print('The shape of the data after converting categorical variables to dummy variables:', data.shape)
    print('The columns of the data after converting categorical variables to dummy variables:', data.columns.values)

    # Splitting dataset into features (X) and target (y)
    X = data.drop('readmitted', axis=1)
    y = data['readmitted']

    # Splitting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature selection with RFE
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    rfe = RFE(estimator=model, n_features_to_select=5)
    rfe.fit(X_train, y_train)

    # Selected features
    selected_features = X.columns[rfe.support_]

    # Fitting model with selected features
    model.fit(X_train[selected_features], y_train)

    # Evaluate the model with K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train[selected_features], y_train, cv=kf)

    # Performance metrics
    y_pred = model.predict(X_test[selected_features])
    overall_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred)
    }

    # Output the results
    return {
        "Selected Features": selected_features.tolist(),
        "Cross-Validation Score": cv_scores.mean(),
        "Overall Metrics": overall_metrics
    }


def balance_data_oversampling(data):
    df_majority = data[data.readmitted==0]
    df_minority = data[data.readmitted==1]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     
                                     n_samples=len(df_majority),   
                                     random_state=123)
    
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    # Display new class counts
    print(df_upsampled.readmitted.value_counts())
    
    return df_upsampled

def process_data_v2(data):
    print("Shape of the data:\n", data.shape)
    
    data.drop_duplicates(subset=['patient_nbr'], keep='last', inplace=True)
    print(data.shape)
    
    columns_to_delete = [
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'payer_code', 'weight', 'medical_specialty', 'encounter_id', 'max_glu_serum']
    data.drop(columns=columns_to_delete, inplace=True)
    print(data.shape)

    data.replace('?', np.nan, inplace=True)
    data.replace('Unknown/Invalid', np.nan, inplace=True)
    # Fetching all the values of the columns
    data['discharge_disposition_id'].unique()
    
    not_alive_patients = data[data['discharge_disposition_id'].isin([11, 19, 20, 21])].index
    # Dropping the identified rows in place
    data.drop(index=not_alive_patients, inplace=True)
    print(data.shape)

    # Remapping it again to 0 and 1
    data['readmitted'] = data['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})
    
    data['gender'].value_counts()

    # # Dropping since only 3 rows have unknown
    data.drop(data[data['gender']=='Unknown/Invalid'].index, inplace=True)
    print(data.shape)

    # Changing age to the mean value instead of the range
    data.loc[(data[data['age'] =='[0-10)'].index), 'age'] = 5
    data.loc[(data[data['age'] =='[10-20)'].index), 'age'] = 15
    data.loc[(data[data['age'] =='[20-30)'].index), 'age'] = 25
    data.loc[(data[data['age'] =='[30-40)'].index), 'age'] = 35
    data.loc[(data[data['age'] =='[40-50)'].index), 'age'] = 45
    data.loc[(data[data['age'] =='[50-60)'].index), 'age'] = 55
    data.loc[(data[data['age'] =='[60-70)'].index), 'age'] = 65
    data.loc[(data[data['age'] =='[70-80)'].index), 'age'] = 75
    data.loc[(data[data['age'] =='[80-90)'].index), 'age'] = 85
    data.loc[(data[data['age'] =='[90-100)'].index), 'age'] = 95


    data['race'].dropna(inplace=True)
    print(data.shape)

    # Draw box plot for numerical columns
    numerical_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    # for col in numerical_cols:
    #     sns.boxplot(data[col])
    #     plt.show()
    
    print(data.shape)

    print(data.columns.values)

    for column in ['diag_1', 'diag_2', 'diag_3']:
        # Ensure the column is of type string for string operations
        data[column] = data[column].astype(str)
        
        # Apply the categorization logic to each column
        data[column] = np.select(
            [
                data[column].str.contains('V') | data[column].str.contains('E'),
                data[column].str.contains('250'),
                ((pd.to_numeric(data[column], errors='coerce').between(390, 459)) | (pd.to_numeric(data[column], errors='coerce') == 785)),
                ((pd.to_numeric(data[column], errors='coerce').between(460, 519)) | (pd.to_numeric(data[column], errors='coerce') == 786)),
                ((pd.to_numeric(data[column], errors='coerce').between(520, 579)) | (pd.to_numeric(data[column], errors='coerce') == 787)),
                ((pd.to_numeric(data[column], errors='coerce').between(580, 629)) | (pd.to_numeric(data[column], errors='coerce') == 788)),
                pd.to_numeric(data[column], errors='coerce').between(140, 239),
                pd.to_numeric(data[column], errors='coerce').between(710, 739),
                pd.to_numeric(data[column], errors='coerce').between(800, 999),
            ], 
            [
                'Other', 'Diabetes', 'Circulatory', 'Respiratory', 
                'Digestive', 'Genitourinary', 'Neoplasms', 
                'Musculoskeletal', 'Injury'
            ], 
            default='Other'
        )
    print("Before removing outliers:", data.shape)
    outlier_cols = ['time_in_hospital', 'num_procedures', 'num_medications', 'number_diagnoses', 'num_lab_procedures']
    data = remove_outliers(data, outlier_cols)
    print("After removing outliers:", data.shape)
    normalization_columns = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses', 'number_outpatient', 'number_emergency', 'number_inpatient']
    data = feature_normalization(data, normalization_columns)

    # Transform the categorical columns into dummy variables
    cat_cols = ['race', 'gender', 'age', 'admission_type_id' , 'discharge_disposition_id', 'admission_source_id', 'diag_1',
 'diag_2', 'diag_3', 'A1Cresult', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed']
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    return data

# Function to determine optimal clusters using the Elbow Method
def determine_optimal_clusters(df, max_k=10):
    ssd = []  # Sum of squared distances
    range_k = range(1, max_k + 1)
    for k in range_k:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        ssd.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range_k, ssd, 'bx-')
    plt.xlabel('k (Number of Clusters)')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

# Function to perform K-Means clustering and visualize the results
def cluster_and_visualize(df, n_clusters):
    # Assuming df is ready for clustering (i.e., numerical and normalized)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df)
    
    df['Cluster'] = clusters  # Append cluster assignments to df
    
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df.drop('Cluster', axis=1))
    
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters  # Add cluster information for plotting
    
    # Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='bright', legend='full')
    plt.title('Clusters Visualized After PCA Reduction')
    plt.show()

    return df

def main():
    data = pd.read_csv('diabetic_data.csv')
    data = process_data(data)
    print('The shape of the data after processing:', data.shape)
    target_var = ['readmitted']
    print('The numerical cols are:', data.select_dtypes(include='number').columns.values)
   
    # Not removing outliers from these columns since they are categorical types

    categorical_int_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
   
   # Not removing outliers from these columns since values are inside 3 standard deviations
    non_outlier_cols = ['number_outpatient', 'number_emergency', 'number_inpatient', 'time_in_hospital', 'num_procedures']

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
    print(data.columns.values)

    # Data Visualisation
    data_visualisation(data, categorical_int_cols)
    
    # Evaluate model performance
    results = evaluate_model_performance(data)
    print(results)

    # Perform oversampling to balance the data
    data_balanced = balance_data_oversampling(data)    
    print('The shape of the balanced data:', data_balanced.shape)
    
    # Evaluate model performance after balancing the data
    results = evaluate_model_performance(data_balanced)
    print(results)

    # data.to_csv('processed_data.csv', index=False)

    # Building a better model

    data_v2 = pd.read_csv('diabetic_data.csv')
    data_v2 = process_data_v2(data_v2)

    features_for_clustering = data_v2.drop(['readmitted'], axis=1)  # Exclude target variable if exists
    
    # Determine the optimal number of clusters
    determine_optimal_clusters(features_for_clustering, max_k=10)
    
    n_clusters = int(input("Enter the optimal number of clusters based on the elbow plot: "))
    
    # Perform K-Means clustering and visualize
    clustered_data = cluster_and_visualize(features_for_clustering, n_clusters)

    # Further analysis of clustered_data can be performed here


if __name__ == '__main__':
    main()
