import random

import numpy as np
import pandas as pd
import preprocessing as pp
import skops.io as skops

random.seed(42)
np.random.seed(42)


def infer():
    df_test = pd.read_csv(
        """https://raw.githubusercontent.com/
        Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"""
    )

    print("Test data shape: ", df_test.shape)

    # Очищаем столбцы от единиц измерения и приводим к float
    df_test = pp.mileage_clean_n_cast(df_test)
    df_test = pp.engine_clean_n_cast(df_test)
    df_test = pp.max_power_clean_n_cast(df_test)

    # Столбец torque просто дропнем
    df_test = df_test.drop(columns="torque")

    # Заполняем пустые значения медианами
    df_test = pp.filling_na_with_medians(df_test, "infer")

    # Кастуем к инту
    df_test["engine"] = df_test["engine"].astype("int64")
    df_test["seats"] = df_test["seats"].astype("int64")

    # Дроп ненужных столбцов
    if "selling_price" in df_test.columns:
        X_test = df_test.drop(columns=["selling_price", "name"])
    else:
        X_test = df_test.drop(columns=["name"])

    # стандартизируем вещественные признаки
    X_test_num_scaled = pp.normalize_numerical(X_test, "infer")

    # кодируем категориальные признаки
    X_test_cat_coded = pp.encoding_categorical(X_test, "infer")

    # объединяем стандартизованный и закодированный элементы
    X_test_preprocessed = X_test_num_scaled.merge(
        X_test_cat_coded, how="inner", left_index=True, right_index=True
    )
    X_test_preprocessed = X_test_preprocessed.to_numpy()

    # Импорт модели
    # ridge_reg = skops.load("main_model/ridge.skops", trusted=False)
    ridge_reg = skops.load("main_model/ridge.skops", trusted=True)

    # Получаем предсказания модели и сохраняем их
    y_pred = ridge_reg.predict(X_test_preprocessed)
    pd.Series(y_pred).to_csv("y_pred.csv")


if __name__ == "__main__":
    infer()
