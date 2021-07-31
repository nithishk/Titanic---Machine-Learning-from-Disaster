import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


train_Data = "Data/train.csv"
train_df = pd.read_csv(train_Data, sep=',', dtype= 'unicode')

print(train_df.shape)

test_Data = "Data/test.csv"
test_df = pd.read_csv(test_Data, sep=',', dtype = 'unicode')

print(test_df.shape)

