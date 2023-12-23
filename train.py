import random

import numpy as np
import pandas as pd
import skops.io as skops
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score

import preprocessing as pp


random.seed(42)
np.random.seed(42)


def train():
    df_train = pd.read_csv(
        """https://raw.githubusercontent.com/
        Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"""
    )

    print("Train data shape:", df_train.shape)

    # Оставляем уникальные объекты
    df_train = pp.train_drop_duplicates(df_train)

    # Очищаем столбцы от единиц измерения и приводим к float
    df_train = pp.mileage_clean_n_cast(df_train)
    df_train = pp.engine_clean_n_cast(df_train)
    df_train = pp.max_power_clean_n_cast(df_train)

    # Столбец torque просто дропнем
    df_train = df_train.drop(columns="torque")

    # Заполняем пустые значения медианами
    df_train = pp.filling_na_with_medians(df_train, "train")

    # Кастуем часть признаков
    df_train["engine"] = df_train["engine"].astype("int64")
    df_train["seats"] = df_train["seats"].astype("int64")

    # вынесем в отдельные переменные таргет
    y_train = df_train["selling_price"].to_numpy()
    X_train = df_train.drop(columns=["selling_price", "name"])

    # стандартизируем вещественные признаки
    X_train_num_scaled = pp.normalize_numerical(X_train, "train")

    # кодируем категориальные признаки
    X_train_cat_coded = pp.encoding_categorical(X_train, "train")

    # объединяем стандартизованный и закодированный элементы
    X_train_preprocessed = X_train_num_scaled.merge(
        X_train_cat_coded, how="inner", left_index=True, right_index=True
    )
    X_train_preprocessed = X_train_preprocessed.to_numpy()

    # Обучение
    ridge_reg = Ridge(alpha=506)
    ridge_reg.fit(X_train_preprocessed, y_train)
    skops.dump(ridge_reg, "main_model/ridge.skops")

    # Оценка модели
    y_pred = ridge_reg.predict(X_train_preprocessed)

    qual_metrics = {
        "r2_train": r2_score(y_train, y_pred),
        "mse_train": MSE(y_train, y_pred),
    }
    [print(f"{k}: {qual_metrics[k]}") for k in qual_metrics]


if __name__ == "__main__":
    train()
