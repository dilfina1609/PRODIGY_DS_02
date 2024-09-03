import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv(r'C:\Users\Dilfina\Downloads\titanic\train.csv')

# Data Cleaning
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

if 'Cabin' in train_df.columns:
    train_df = train_df.drop(columns=['Cabin'])

# Feature Engineering
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop non-numeric columns before correlation
numeric_df = train_df.select_dtypes(include=['number'])

# Exploratory Data Analysis (EDA)
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Survival rate by Pclass
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Pclass')
plt.show()

# Survival rate by Sex
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Sex')
plt.show()

# Distribution of Age
sns.histplot(train_df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Survival rate by Family Size
sns.barplot(x='FamilySize', y='Survived', data=train_df)
plt.title('Survival Rate by Family Size')
plt.show()