import pandas as pd
import numpy as np

# Load the Titanic dataset
train_df = pd.read_csv(r'C:\Users\Dilfina\Downloads\titanic\train.csv')

# 1. Handling Missing Data
# Check for missing values
missing_values = train_df.isnull().sum()
print("Missing values before cleaning:\n", missing_values)

# Drop 'Cabin' column due to many missing values
train_df = train_df.drop('Cabin', axis=1)

# Fill missing values in 'Age' with the median age
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# Fill missing 'Embarked' with the most common value
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Check for missing values after cleaning
missing_values_after = train_df.isnull().sum()
print("\nMissing values after cleaning:\n", missing_values_after)

# 2. Remove Duplicates
train_df = train_df.drop_duplicates()

# 3. Convert Data Types
train_df['Pclass'] = train_df['Pclass'].astype('category')
train_df['Survived'] = train_df['Survived'].astype('category')
train_df['Sex'] = train_df['Sex'].astype('category')
train_df['Embarked'] = train_df['Embarked'].astype('category')

# 4. Handle Outliers (example using 'Fare')
# Cap Fare at the 99th percentile
fare_cap = train_df['Fare'].quantile(0.99)
train_df['Fare'] = np.where(train_df['Fare'] > fare_cap, fare_cap, train_df['Fare'])

# 5. Normalize/Standardize Data (example for 'Fare')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df['Fare'] = scaler.fit_transform(train_df[['Fare']])

# Display the cleaned dataset
print(train_df.head())

