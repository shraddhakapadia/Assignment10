#2

import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('/Users/shraddhakapadia/Documents/BigData/Assignment10/covidtesting.csv')


# Display the first few rows of the dataset to get an overview
print(df.head())

# Get summary statistics of the numerical columns
print(df.describe())

# Get information about data types and non-null values
print(df.info())



#3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset into a pandas DataFrame
df = pd.read_csv('/Users/shraddhakapadia/Documents/BigData/Assignment10/covidtesting.csv')

# Check for missing values in each column
print(df.isnull().sum())

# #Option 1: Remove rows with missing values (if the missing values are relatively small)
df_cleaned = df.dropna()

# Impute missing values with mean
mean_value = df['Confirmed Positive'].mean()
df['Confirmed Positive'].fillna(mean_value, inplace=True)
print(df['Confirmed Positive'])


# Visualize the distribution of a numerical column to identify outliers
sns.boxplot(x='newly_reported_deaths', data=df)
plt.show()

# Remove outliers
q1 = df['Confirmed Positive'].quantile(0.25)
q3 = df['Confirmed Positive'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df_cleaned = df[(df['Confirmed Positive'] >= lower_bound) & (df['Confirmed Positive'] <= upper_bound)]
print(df_cleaned)
# Option 2: Transform outliers 
df['Confirmed Positive'] = np.log1p(df['Confirmed Positive'])
print(df['Confirmed Positive'])

# Perform feature engineering based on your domain knowledge and the nature of the data
# For example, extract new features from existing ones, encode categorical variables, etc.

# Example: Convert a categorical variable to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Total LTC HCW Deaths'], drop_first=True)
print(df_encoded.info())
# Example: Create new features by combining existing ones
df['Total aggregated patients'] = df['Total patients approved for testing as of Reporting Date'] * df['Total Cases']
print(df['Total aggregated patients'])


#4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset into a pandas DataFrame
df = pd.read_csv('/Users/shraddhakapadia/Documents/BigData/Assignment10/covidtesting.csv')

# Drop columns
df_numeric = df.drop(['Confirmed Negative', 'Confirmed Positive', 'Resolved'], axis=1)
X = df.drop('Total patients approved for testing as of Reporting Date', axis=1)
y = df['Reported Date']

X = df.drop('Total patients approved for testing as of Reporting Date', axis=1)  # Features (input variables)
y = df['Reported Date']  # Target variable (output)

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(y_train)

# Create and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
svm_predictions = svm_model.predict(X_test)
print(svm_predictions)
# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test)
print(rf_predictions)

#5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/Users/shraddhakapadia/Documents/BigData/Assignment10/covidtesting.csv')

# Scatter Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x="Confirmed Negative", y="Confirmed Positive", data=df)
plt.title("Scatter Plot: Confirmed Negative vs. Confirmed Positive'")
plt.xlabel("Confirmed Negative")
plt.ylabel("Confirmed Positive'")
plt.show()

#Bar Chart
plt.figure(figsize=(6, 4))
sns.countplot(x="Confirmed Negative", data=df)
plt.title("Total Confirmed Negative Counts")
plt.xlabel("Total Covid Cases")
plt.ylabel("Count")
plt.show()

# Pair Plot
sns.pairplot(df, hue="Total Cases", vars=["Confirmed Negative", "Presumptive Negative", "Presumptive Positive", "Confirmed Positive", "Resolved"])
plt.suptitle("Pair Plot of Key Features with Hue based on Total Cases", y=1.02)
plt.show()

# Box Plot
plt.figure(figsize=(6, 3))
sns.boxplot(x="Total tests completed in the last day", y="Percent positive tests in last day", data=df)
plt.title("Percent positive tests in last day by Total tests completed in the last day")
plt.xlabel("Total tests completed in the last day")
plt.ylabel("Percent positive tests in last day")
plt.show()
