# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data import
train_data = pd.read_csv('data/train.csv')
seq = train_data.loc[:, 'protein_sequence']
lens = [len(s) for s in seq]

outlier = max(range(len(lens)), key=lens.__getitem__)
lens.pop(outlier)
max(lens)
plt.hist(lens)
plt.show()


