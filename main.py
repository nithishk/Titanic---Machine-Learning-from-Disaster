import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

import seaborn as sns

train_Data = "Data/train.csv"
train_df = pd.read_csv(train_Data, sep=',', index_col='PassengerId')

print(train_df.head())

test_Data = "Data/test.csv"
test_df = pd.read_csv(test_Data, sep=',', index_col='PassengerId')

print(train_df.Parch.unique())

Y_test = pd.read_csv("Data/gender_submission.csv", index_col='PassengerId')

print(test_df.info())
print(train_df.info())
print(train_df.describe(include = 'all'))
print(train_df.isnull().sum())

# Data Cleaning & Encoding


# Data visualization
msno.matrix(train_df)
#plt.show()


sns.barplot(x = "Sex", y = "Survived", data = train_df)

print("Percentage of females who survived: ", train_df["Survived"][train_df["Sex"] == 'female'].value_counts(normalize=True)[1]*100)
print("Percentage of males who survived:", train_df["Survived"][train_df["Sex"] == 'male'].value_counts(normalize= True)[1]*100)









