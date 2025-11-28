import numpy as np
import pandas as pd

# Exploration of the dataset

df = pd.read_csv('heart.csv')

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Columns:")
print(df.columns)    