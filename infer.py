import random
import skops.io as skops

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder 
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error as MSE

import preprocessing as pp

random.seed(42)
np.random.seed(42)

df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

print("Test data shape: ", df_test.shape)

df_test = pp.mileage_clean_n_cast(df_test)
df_test = pp.engine_clean_n_cast(df_test)
df_test = pp.max_power_clean_n_cast(df_test)

df_test = df_test.drop(columns='torque')

df_test = pp.filling_na_with_medians(df_test)

df_test['engine'] = df_test['engine'].astype('int64')
df_test['seats'] = df_test['seats'].astype('int64')
#------------------------------------------------------------------------------------
# y_test = df_test['selling_price'].to_numpy()
# X_test = df_test.drop(columns=['selling_price', 'name'])

# X_test_num = X_test[numerical_cols].to_numpy()
# X_test_cat = X_test[categorial_cols].to_numpy()

# X_test_num_scaled = normalizer.transform(X_test)

# # И не забываем закодировать категориальные признаки на тесте + добавить заскейленные вещественные!
# df_test = df_test.drop(columns='name')
# df_test_coded = pd.DataFrame(enc.transform( df_test[df_test.columns[df_test.dtypes == object]] ).toarray(), columns=enc.get_feature_names_out())

# # Уберём лишний столбец
# df_test_coded = df_test_coded[df_test_coded.columns[:-1]]
# df_test_coded.head()

# df_test_coded = pd.concat([
#                           df_test[['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']],
#                           df_test_coded], axis=1)

# X_test_scaled_full_features = pd.DataFrame(X_test_scaled).merge(df_test_coded[df_test_coded.columns[6:]], how='inner', left_index=True, right_index=True).to_numpy()