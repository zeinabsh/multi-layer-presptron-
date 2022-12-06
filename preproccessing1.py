
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('penguins.csv')
####################################################################################

# _________________ Data Preprocessing____________________

# Removing null values
df['gender'].fillna('male', inplace=True)
# male replaced with 1
df['gender'].replace('male', 1, inplace=True)
# female replaced with 0
df['gender'].replace('female', 0, inplace=True)

# normalized data
columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'gender']
for column in columns:
    df[column] = MinMaxScaler().fit_transform(np.array(df[column]).reshape(-1, 1))

df['species'].replace('Adelie', 1, inplace=True)
df['species'].replace('Gentoo', -1, inplace=True)
df['species'].replace('Chinstrap', 2, inplace=True)

df['Adelie'] = np.where(df['species'] == 1, 1, 0)
df['Gentoo'] = np.where(df['species'] == -1, 1, 0)
df['Chinstrap'] = np.where(df['species'] == 2, 1, 0)

df_Adelie = df[df['species'] == 1]
df_Gentoo = df[df['species'] == -1]
df_Chinstrap = df[df['species'] == 2]

feature1 = 'bill_length_mm'
feature2 = 'bill_depth_mm'
feature3 = 'flipper_length_mm'
feature4 = 'gender'
feature5 = 'body_mass_g'
#####################################################################################################

# _________split_data____________
all_train = pd.concat([df_Adelie[:30], df_Gentoo[:30], df_Chinstrap[:30]])
all_train = all_train.sample(frac=1).reset_index(drop=True)

all_test = pd.concat([df_Adelie[30:], df_Gentoo[30:], df_Chinstrap[30:]])
all_test = all_test.sample(frac=1).reset_index(drop=True)

X_train = all_train[[feature1, feature2, feature3, feature4, feature5]]
y_train = all_train[['Adelie', 'Gentoo', 'Chinstrap']].astype(float)

X_test = all_test[[feature1, feature2, feature3, feature4, feature5]]
y_test = all_test[['Adelie', 'Gentoo', 'Chinstrap']].astype(float)
