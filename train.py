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

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
# df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

print("Train data shape:", df_train.shape)
# print("Test data shape: ", df_test.shape)

# Оставляем уникальные объекты
df_train = pp.train_drop_duplicates(df_train)

# Очищаем столбцы от единиц измерения и приводим к float
df_train = pp.mileage_clean_n_cast(df_train)
df_train = pp.engine_clean_n_cast(df_train)
df_train = pp.max_power_clean_n_cast(df_train)
# df_train, df_test = pp.mileage_clean_n_cast(df_train, df_test)
# df_train, df_test = pp.engine_clean_n_cast(df_train, df_test)
# df_train, df_test = pp.max_power_clean_n_cast(df_train, df_test)

# Столбец torque просто дропнем
df_train = df_train.drop(columns='torque')
# df_test = df_test.drop(columns='torque')

# Заполняем пустые значения медианами
df_train = pp.filling_na_with_medians(df_train)
# df_train, df_test = pp.filling_na_with_medians(df_train, df_test)

# Кастуем часть признаков
df_train['engine'] = df_train['engine'].astype('int64')
# df_test['engine'] = df_test['engine'].astype('int64')
df_train['seats'] = df_train['seats'].astype('int64')
# df_test['seats'] = df_test['seats'].astype('int64')


# ----------------
# --- training ---
# ----------------

# вынесем в отдельные переменные таргет
y_train = df_train['selling_price'].to_numpy()
X_train = df_train.drop(columns=['selling_price', 'name'])

# ---ВЕЩЕСТВЕННЫЕ ПРИЗНАКИ---
# Стандартизуем вещественные признаки
numerical_cols = X_train.columns[X_train.dtypes != object]
X_train_num = X_train[numerical_cols].to_numpy()

normalizer = StandardScaler()
normalizer.fit(X_train_num)
X_train_num_scaled = normalizer.transform(X_train_num)
X_train_num_scaled = pd.DataFrame(X_train_num_scaled,
                                  columns=numerical_cols)

# ---КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ---
# Кодируем OHE
categorial_cols = X_train.columns[X_train.dtypes == object]
X_train_cat = X_train[categorial_cols].to_numpy()

enc = OneHotEncoder()
enc.fit(X_train_cat)
X_train_cat_coded = enc.transform(X_train_cat).toarray()
X_train_cat_coded = pd.DataFrame(X_train_cat_coded,
                                 columns=enc.get_feature_names_out())
# Уберём лишний столбец после энкодинга
X_train_cat_coded = X_train_cat_coded[X_train_cat_coded.columns[:-1]]

# Объединяем
X_train_scaled_full_features = X_train_num_scaled.merge(X_train_cat_coded,
                                                        how='inner', 
                                                        left_index=True,
                                                        right_index=True)

X_train_scaled_full_features = X_train_scaled_full_features.to_numpy()

# Обучение
# ridge_reg = Ridge(alpha=506)
# ridge_reg.fit(X_train_scaled_full_features, y_train)
# skops.dump(ridge_reg, "ridge.skops")
ridge_reg = skops.load("ridge.skops")

pred_train_l2 = ridge_reg.predict(X_train_scaled_full_features)

r2_scores_l2_opt = pd.Series({'r2_train': r2_score(y_train, pred_train_l2)})
mses_l2_opt = pd.Series({'mse_train': MSE(y_train, pred_train_l2)})

print(r2_scores_l2_opt, '\n')
print(mses_l2_opt)
