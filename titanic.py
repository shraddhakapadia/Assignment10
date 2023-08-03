#!/usr/bin/env python
# coding: utf-8

# In[32]:


#2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/Users/shraddhakapadia/Documents/BigData/Assignment10/titanic.csv')

# Display the first few rows of the dataset to get an overview
print(df.head())

# Get summary statistics of the numerical columns
print(df.describe())

# Get information about data types and non-null values
print(df.info())

# plotting a histogram for a numerical column 'column_name'
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()





#3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Handle Missing Values:
# load the dataset
df = pd.read_csv('/Users/shraddhakapadia/Documents/BigData/Assignment10/titanic.csv')

# Check for missing values in each column
print(df.isnull().sum())

# Remove rows with missing values
df_cleaned_SK = df.dropna()
print(df_cleaned_SK)

# missing values with mean
mean_value = df['Fare'].mean()
print(mean_value)
df['Fare'].fillna(mean_value, inplace=True)
		
#Handle Outliers:
# Visualize the distribution of a Fare to identify outliers
sns.boxplot(x='Fare', data=df)
plt.show()

# Remove outliers (if they are extreme and few in number)
q1 = df['Fare'].quantile(0.25)
q3 = df['Fare'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
df_cleaned_SK = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]
print(df_cleaned_SK)

# Transform outliers
df['Fare'] = np.log1p(df['Fare'])
print(df['Fare'])
		
#Feature Engineering:
# Perform feature engineering based on your domain knowledge and the nature of the data
# For example, extract new features from existing ones, encode categorical variables, etc.

# Convert a categorical variable to numerical using one-hot encoding
df_encoded_SK= pd.get_dummies(df, columns=['Name'], drop_first=True)
print(df_encoded_SK)

# Create new features by combining existing ones
df['Name'] = df['Age'] * df['Survived']
print(df['Name'])



#4
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Loaded the dataset
df = pd.read_csv('/Users/shraddhakapadia/Documents/BigData/Assignment10/titanic.csv')
print(df)

# Data Preprocessing
# Convert 'Sex' column to numeric representation (1 for male, 0 for female)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# Fill missing values in 'Age' column with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# One-hot encoding for 'Embarked' column
le = LabelEncoder()
df['Embarked'] = le.fit_transform(df['Embarked'])

# Drop 'Name', 'Ticket', and 'Cabin' columns
df_numeric = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into features (X) and target (y)
X = df_numeric.drop('Survived', axis=1)
y = df_numeric['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

svm_model = SVC(kernel='linear', random_state=1)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of SVM: {svm_accuracy:.2f}')
classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)

random_forest_model = RandomForestClassifier(random_state=1)
random_forest_model.fit(X_train, y_train)

y_pred = random_forest_model.predict(X_test)
frst_accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Random Forest: {frst_accuracy:.2f}')

classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)


#5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/Users/shraddhakapadia/Documents/BigData/Assignment10/titanic.csv')

# Scatter Plot: Age vs. Fare
plt.figure(figsize=(6, 4))
sns.scatterplot(x="Age", y="Fare", data=df)
plt.title("Scatt	er Plot: Age vs. Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# Bar Chart: Survival Count
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# Pair Plot
sns.pairplot(df, hue="Survived", vars=["Age", "Fare", "Pclass", "SibSp", "Parch"])
plt.suptitle("Pair Plot of Key Features with Hue based on Survival", y=1.02)
plt.show()

# Box Plot: Fare by Pclass
plt.figure(figsize=(8, 6))
sns.boxplot(x="Pclass", y="Fare", data=df)
plt.title("Fare by Pclass")
plt.xlabel("Pclass")
plt.ylabel("Fare")
plt.show()


# In[ ]:
