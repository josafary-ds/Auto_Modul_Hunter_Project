import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

# Charging feets file
file = pd.read_csv('feets_full_table.csv', sep=';')

# Dividing columns into predictors and class
x_feets = file.iloc[:, 2:14].values
y_feets = file.iloc[:, 14].values

# Verifying infinity values and replacing to NaN
if np.isinf(x_feets).sum() != 0:
    x_feets = np.where(np.isinf(x_feets), np.nan, x_feets)
else:
    pass

# Treating NaN values and scaling it
imputer = SimpleImputer(strategy='median')
x_feets = imputer.fit_transform(x_feets)

scaler_feets = StandardScaler()
x_feets = scaler_feets.fit_transform(x_feets)

# Applying LabelEncoder to class column
y_feets = LabelEncoder().fit_transform(y_feets)

# Creating training-test basis
x_feets_training, x_feets_test, y_feets_training, y_feets_test = train_test_split(x_feets, y_feets, test_size=0.25, random_state=0)

# Saving into pkl file
with open('feets_data.pkl', mode='wb') as f:
    pickle.dump([x_feets_training, x_feets_test, y_feets_training, y_feets_test], f)