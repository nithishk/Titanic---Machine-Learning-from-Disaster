import numpy as np
from  sklearn.tree import DecisionTreeClassifier
import pandas as pd
Data = "Data/train.csv"
df = pd.read_csv(Data, sep=',', dtype= 'unicode')
missing = df.isnull().sum()
print(missing)
df.dropna(how='all')
print(df.info())

